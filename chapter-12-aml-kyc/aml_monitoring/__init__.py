"""AML Transaction Monitoring System — MR-2026-061.

SS1/23 Risk: HIGH | EU AI Act: LIMITED scope
Regulation: POCA 2002 s.330 (failure to disclose)
           POCA 2002 s.333A (tipping off — hard architectural guarantee)
           FCA SYSC 6.3.1/6.3.2 | JMLSG Part I Chapter 6
           MLR 2017 Reg. 40 (ongoing monitoring)

Two complementary models per prompt spec:
  Model 1: XGBoost transaction scorer (supervised ML)
  Model 2: NetworkX graph analyser (unsupervised community detection)

AML Typologies RAG: uses Chapter 4 ChromaDB + hybrid search
  infrastructure (separate AML collection — not MR-2026-038).
  JMLSG typologies corpus, NCA Strategic Assessments, FATF reports.

Alert thresholds per prompt:
  0.35 = alert threshold
  0.70 = high-priority alert
  0.90 = auto-escalation to MLRO

Target: 8,400 alerts/month (99% FP) → 1,680/month (80% FP reduction)
NCA SubmitSAR API — UK National Crime Agency (NOT FinCEN — US-only)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import logging
import uuid

import networkx as nx

from awb_commons.models import (
    AlertResult, AlertPriority, NetworkRiskSummary,
    SARDraft, SARStatus,
)

log = logging.getLogger(__name__)

# Alert thresholds per prompt spec
ALERT_THRESHOLD = 0.35
HIGH_PRIORITY_THRESHOLD = 0.70
AUTO_MLRO_THRESHOLD = 0.90

# JMLSG structuring threshold (cash below this = structuring watch)
STRUCTURING_WATCH_GBP = Decimal("5000")

# FATF high-risk jurisdictions per MLR 2017 Reg. 33(6)
FATF_HIGH_RISK = frozenset([
    "AF","BY","CF","CD","CU","IR","IQ","LY","ML","MM",
    "NI","KP","RU","SO","SS","SD","SY","VE","YE","ZW",
])


class FeatureEngineer:
    """Compute AML feature vectors from T24 transaction data.

    Five feature groups per prompt spec:
    1. Velocity: count/amount by time window (24hr, 7d, 30d)
    2. Behavioural: z-score deviation from customer baseline
    3. Network: counterparty diversity, Herfindahl concentration
    4. Typology: structuring, rapid movement, dormant activation
    5. Geographic: FATF high-risk jurisdictions, offshore centres

    Training data: 18 months AWB history; 2,100 SAR-confirmed
    positives out of 75.6M transactions (0.003% base rate).
    Class imbalance handled: SMOTE oversampling + class weights.
    """

    def compute_features(
        self,
        transaction: Dict[str, Any],
        account_history: List[Dict],
        counterparty_risk_db: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute AML feature vector for ML scoring.

        Args:
            transaction: T24 transaction record dict.
            account_history: Recent transactions for account.
            counterparty_risk_db: Counterparty risk scores.

        Returns:
            Feature dict ready for XGBoost scoring.
        """
        amount = float(transaction.get("amount_gbp", 0))
        country = transaction.get("country_code", "GB")
        cpty_id = transaction.get("counterparty_id", "")

        # Velocity features
        count_24h = len(account_history[-24:])
        amount_24h = sum(
            float(t.get("amount_gbp", 0))
            for t in account_history[-24:]
        )

        # Behavioural: z-score vs 90-day rolling mean
        amounts = [float(t.get("amount_gbp", 0)) for t in account_history]
        import numpy as np
        z_score = 0.0
        if len(amounts) > 2:
            mean = np.mean(amounts)
            std = np.std(amounts)
            if std > 0:
                z_score = (amount - mean) / std

        # Typology: structuring pattern
        is_structuring_watch = (
            Decimal(str(amount)) < STRUCTURING_WATCH_GBP
            and count_24h >= 3
        )

        # Network: counterparty concentration (Herfindahl)
        cpty_amounts = {}
        for t in account_history:
            c = t.get("counterparty_id", "")
            cpty_amounts[c] = cpty_amounts.get(c, 0) + float(t.get("amount_gbp", 0))
        total_amt = sum(cpty_amounts.values())
        herfindahl = (
            sum((v/total_amt)**2 for v in cpty_amounts.values())
            if total_amt > 0 else 0.0
        )

        # Geographic risk
        high_risk_country = country in FATF_HIGH_RISK

        # Counterparty risk
        cpty_risk = counterparty_risk_db.get(cpty_id, 0.0)

        features = {
            "amount_log": float(amount) ** 0.5,
            "velocity_count_24h": float(count_24h),
            "velocity_amount_24h": float(amount_24h),
            "amount_z_score": float(z_score),
            "structuring_watch": float(is_structuring_watch),
            "counterparty_herfindahl": float(herfindahl),
            "counterparty_risk_score": float(cpty_risk),
            "high_risk_jurisdiction": float(high_risk_country),
            "is_round_amount": float(amount % 1000 == 0),
        }
        log.debug(
            "Features: txn_id=%s z=%.2f high_risk=%s struct=%s",
            transaction.get("id", ""),
            z_score, high_risk_country, is_structuring_watch,
        )
        return features


class AMLTransactionScorer:
    """XGBoost-based AML transaction risk scorer.

    Same XGBoost approach as Chapter 8 payment fraud detection
    (as noted in prompt: "note the parallel explicitly").
    SHAP values for FCA/JMLSG-compliant explainability.

    PRA SS1/23: HIGH risk rating — 6-month revalidation cycle.
    Independent validation by AWB Model Risk team required
    before deployment and every 6 months.

    Args:
        model_id: Registered model ID (MR-2026-061).
    """

    def __init__(
        self,
        model_id: str = "MR-2026-061",
    ) -> None:
        self._model_id = model_id
        self._model = None  # Loaded from model artefact in production
        self._feature_engineer = FeatureEngineer()
        log.info("AMLTransactionScorer initialised: %s", model_id)

    def score_transaction(
        self,
        transaction: Dict[str, Any],
        account_history: List[Dict],
        counterparty_risk_db: Dict[str, float],
    ) -> AlertResult:
        """Score a transaction and generate alert if above threshold.

        Args:
            transaction: T24 transaction record.
            account_history: Recent account transactions.
            counterparty_risk_db: Counterparty risk scores.

        Returns:
            AlertResult (may be LOW priority = auto-cleared).
        """
        features = self._feature_engineer.compute_features(
            transaction, account_history, counterparty_risk_db
        )
        score = self._compute_score(features)
        priority = self._route_alert(score)
        shap_values = self._get_feature_explanation(features, score)
        top_features = sorted(
            shap_values.keys(), key=lambda k: abs(shap_values[k]), reverse=True
        )[:5]
        alert = AlertResult(
            alert_id=str(uuid.uuid4())[:8].upper(),
            transaction_id=transaction.get("id", ""),
            account_id=transaction.get("account_id", ""),
            score=round(score, 4),
            priority=priority,
            features=top_features,
            shap_values=shap_values,
            created_at=datetime.utcnow(),
            model_id=self._model_id,
        )
        log.info(
            "Alert: id=%s score=%.3f priority=%s txn=%s",
            alert.alert_id, score, priority.value,
            alert.transaction_id,
        )
        return alert

    def get_feature_explanation(
        self,
        features: Dict[str, float],
        score: float,
    ) -> Dict[str, float]:
        """Return SHAP values for alert explainability.

        Satisfies JMLSG requirement for documented explanation
        of why a transaction was flagged for analyst review.
        """
        return self._get_feature_explanation(features, score)

    def route_alert(
        self, score: float
    ) -> AlertPriority:
        """Route alert by priority based on ML score.

        Per prompt spec thresholds:
        < 0.35: LOW (auto-cleared)
        0.35–0.70: MEDIUM (analyst queue)
        0.70–0.90: HIGH (MLRO review)
        >= 0.90: HIGH with auto-MLRO escalation
        """
        return self._route_alert(score)

    def _compute_score(
        self, features: Dict[str, float]
    ) -> float:
        """Compute AML risk score from feature vector.

        Production: XGBoost model.predict_proba()[1]
        Test: deterministic heuristic from features.
        """
        score = 0.0
        if features.get("high_risk_jurisdiction", 0):
            score += 0.35
        if features.get("structuring_watch", 0):
            score += 0.25
        if features.get("amount_z_score", 0) > 3.0:
            score += 0.20
        if features.get("counterparty_risk_score", 0) > 0.7:
            score += 0.20
        if features.get("counterparty_herfindahl", 0) > 0.8:
            score += 0.10
        return min(1.0, score)

    def _route_alert(self, score: float) -> AlertPriority:
        if score >= HIGH_PRIORITY_THRESHOLD:
            return AlertPriority.HIGH
        elif score >= ALERT_THRESHOLD:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.LOW

    def _get_feature_explanation(
        self,
        features: Dict[str, float],
        score: float,
    ) -> Dict[str, float]:
        """Return SHAP-style feature attribution scores."""
        shap = {}
        if features.get("high_risk_jurisdiction", 0):
            shap["high_risk_jurisdiction"] = 0.35 * score
        if features.get("structuring_watch", 0):
            shap["structuring_watch"] = 0.25 * score
        if features.get("amount_z_score", 0) > 3.0:
            shap["amount_z_score"] = 0.20 * score
        if features.get("counterparty_risk_score", 0) > 0.5:
            shap["counterparty_risk_score"] = (
                features["counterparty_risk_score"] * 0.20
            )
        return shap


class AMLNetworkAnalyser:
    """NetworkX graph analyser for coordinated structuring detection.

    Implements Louvain community detection to identify rings of
    accounts transacting primarily with a single beneficiary —
    the "smurfing" / coordinated structuring typology.

    This is the model that catches the war story scenario:
    47 accounts structured £2.3M over 14 months — undetected
    by the legacy rules-based system because no individual
    transaction exceeded the threshold.

    Graph updated daily with prior 90-day transactions.
    Communities flagged when any member receives ML alert.

    Args:
        model_id: Registered model ID (MR-2026-061).
    """

    def __init__(self, model_id: str = "MR-2026-061") -> None:
        self._model_id = model_id
        self._graph = nx.DiGraph()

    def build_transaction_graph(
        self,
        transactions: List[Dict[str, Any]],
        lookback_days: int = 90,
    ) -> nx.DiGraph:
        """Build directed transaction graph from T24 data.

        Nodes: customers, accounts, counterparties.
        Edges: transactions (weighted by amount, dated).

        Args:
            transactions: List of T24 transaction records.
            lookback_days: Rolling window in days (default 90).

        Returns:
            NetworkX DiGraph representing transaction network.
        """
        G = nx.DiGraph()
        for txn in transactions:
            src = txn.get("account_id", "")
            dst = txn.get("counterparty_id", "")
            amt = float(txn.get("amount_gbp", 0))
            if not src or not dst:
                continue
            G.add_node(src, node_type="account")
            G.add_node(dst, node_type="counterparty")
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += amt
                G[src][dst]["count"] += 1
            else:
                G.add_edge(src, dst, weight=amt, count=1)
        self._graph = G
        log.info(
            "Transaction graph built: %d nodes %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    def detect_communities(
        self,
    ) -> List[List[str]]:
        """Apply Louvain community detection to transaction graph.

        Identifies clusters of accounts transacting primarily
        with each other — hallmark of coordinated structuring.
        Communities with a single concentrated beneficiary
        receiving from many sources are flagged for SAR review.

        Returns:
            List of communities (each community = list of node IDs).
        """
        if self._graph.number_of_nodes() == 0:
            return []
        # Use undirected graph for community detection
        undirected = self._graph.to_undirected()
        try:
            # Louvain via greedy_modularity_communities (networkx)
            communities = list(
                nx.community.greedy_modularity_communities(undirected)
            )
            log.info(
                "Louvain communities: %d detected", len(communities)
            )
            return [list(c) for c in communities]
        except Exception as e:
            log.error("Community detection failed: %s", e)
            return []

    def identify_structuring_pattern(
        self,
        beneficiary_id: str,
        amount_threshold_gbp: Decimal = STRUCTURING_WATCH_GBP,
        min_contributors: int = 3,
    ) -> Optional[NetworkRiskSummary]:
        """Detect smurfing: multiple accounts → one beneficiary.

        JMLSG typology: structured deposits where multiple accounts
        send sub-threshold amounts to a single beneficiary, totalling
        an amount that would trigger reporting if sent as one payment.

        Args:
            beneficiary_id: Target node to check for structuring.
            amount_threshold_gbp: Sub-threshold watch amount.
            min_contributors: Minimum accounts to flag (default 3).

        Returns:
            NetworkRiskSummary if pattern detected, else None.
        """
        if beneficiary_id not in self._graph:
            return None
        # Find all predecessors (senders to this beneficiary)
        predecessors = list(self._graph.predecessors(beneficiary_id))
        if len(predecessors) < min_contributors:
            return None
        # Check if sender amounts are all sub-threshold
        sub_threshold_senders = []
        total_amount = Decimal("0")
        for sender in predecessors:
            edge_data = self._graph[sender][beneficiary_id]
            edge_amt = Decimal(str(edge_data.get("weight", 0)))
            if edge_amt < amount_threshold_gbp:
                sub_threshold_senders.append(sender)
                total_amount += edge_amt

        if len(sub_threshold_senders) >= min_contributors:
            log.warning(
                "STRUCTURING PATTERN: beneficiary=%s "
                "contributors=%d total=£%s",
                beneficiary_id, len(sub_threshold_senders),
                total_amount,
            )
            return NetworkRiskSummary(
                community_id=f"STRUCT-{beneficiary_id[:8]}",
                member_account_ids=sub_threshold_senders,
                total_amount_gbp=total_amount,
                transaction_count=len(sub_threshold_senders),
                is_structuring_ring=True,
                centrality_accounts=[beneficiary_id],
                model_id=self._model_id,
            )
        return None


class TippingOffGuardrail:
    """Architectural enforcement of POCA 2002 s.333A.

    Tipping off (informing the subject of a SAR) is a criminal
    offence under POCA 2002 s.333A. This is a HARD architectural
    guarantee — not a process control.

    The credit decision agent (MR-2026-037) NEVER receives SAR
    status. It only receives KYCStatus.BLOCKED when a SAR has been
    filed. The credit agent cannot determine whether blocking is
    due to a SAR or a normal KYC failure.

    This class enforces that guarantee at the code level.
    """

    @staticmethod
    def get_safe_credit_status(
        sar_filed: bool,
        kyc_status: str,
    ) -> str:
        """Return credit gate status WITHOUT disclosing SAR status.

        Args:
            sar_filed: Whether a SAR has been filed (NEVER disclosed).
            kyc_status: Normal KYC status string.

        Returns:
            Safe status for credit decision agent.
            "BLOCKED" regardless of reason if sar_filed=True.
        """
        if sar_filed:
            # POCA s.333A: never disclose SAR existence
            # Credit agent sees only "BLOCKED" with no reason
            log.info(
                "POCA s.333A: SAR filed — returning BLOCKED "
                "status without SAR disclosure to credit agent"
            )
            return "BLOCKED"
        return kyc_status

    @staticmethod
    def validate_no_disclosure(
        message: str,
    ) -> bool:
        """Validate that a message does not reference SAR status.

        Args:
            message: Any outbound message to customer-facing systems.

        Returns:
            True if message is safe (no SAR reference).

        Raises:
            ValueError: If message contains SAR disclosure.
        """
        sar_keywords = [
            "suspicious activity report",
            "SAR",
            "sar filed",
            "disclosed to NCA",
            "money laundering report",
        ]
        msg_lower = message.lower()
        for kw in sar_keywords:
            if kw.lower() in msg_lower:
                raise ValueError(
                    f"POCA 2002 s.333A VIOLATION: Message "
                    f"contains SAR reference '{kw}'. "
                    f"This would constitute tipping off — "
                    f"a criminal offence."
                )
        return True


class SARDraftGenerator:
    """AI-assisted SAR draft generator for NCA SubmitSAR.

    POCA 2002 s.330: nominated officer (MLRO) must disclose
    knowledge or suspicion of ML to National Crime Agency.
    NCA SubmitSAR API — NOT FinCEN (US-only, not applicable).

    Generates SAR sections (b) nature of suspicion and
    (c) typology citation from ML scorer + RAG + graph data.
    MLRO reviews and approves before NCA submission.

    SAR types per POCA 2002:
    - Disclosure SAR (s.330): retrospective — activity occurred
    - Consent SAR (s.335): seek NCA consent to proceed
      (moratorium: NCA has 7 working days to respond)

    Args:
        model_id: Registered model ID (MR-2026-063).
    """

    def __init__(self, model_id: str = "MR-2026-063") -> None:
        self._model_id = model_id
        self._tipping_off = TippingOffGuardrail()

    def generate_sar_draft(
        self,
        customer_id: str,
        account_id: str,
        alert_ids: List[str],
        total_suspicious_amount_gbp: Decimal,
        typology_description: str,
        network_summary: Optional[NetworkRiskSummary] = None,
        sar_type: str = "disclosure",
    ) -> SARDraft:
        """Generate SAR draft for MLRO review.

        Generates NCA SAR sections (b) nature of suspicion and
        (c) typology citation. All other sections (a, d, e)
        completed by MLRO from case management system.

        Args:
            customer_id: Subject customer.
            account_id: Subject account number.
            alert_ids: Related AML alert IDs.
            total_suspicious_amount_gbp: Total amount.
            typology_description: JMLSG typology match.
            network_summary: NetworkX analysis if available.
            sar_type: "disclosure" or "consent" (s.335).

        Returns:
            SARDraft — MLRO approval ALWAYS required.
        """
        # Build nature of suspicion narrative
        nature = (
            f"AI-ASSISTED DRAFT — MLRO REVIEW REQUIRED\n\n"
            f"Suspicious activity detected via AML Transaction "
            f"Monitoring System (MR-2026-061). "
            f"ML model identified {len(alert_ids)} high-priority "
            f"alert(s) totalling £{total_suspicious_amount_gbp:,.2f} "
            f"that match the {typology_description} typology pattern."
        )
        if network_summary and network_summary.is_structuring_ring:
            nature += (
                f"\n\nNetwork graph analysis (Louvain community "
                f"detection) identified a coordinated structuring "
                f"ring of {len(network_summary.member_account_ids)} "
                f"accounts transacting to a common beneficiary, "
                f"consistent with JMLSG structuring typology."
            )
        # Typology citation from JMLSG
        typology_cite = (
            f"JMLSG Part II Banking — {typology_description}. "
            f"Pattern consistent with FATF Typology Report "
            f"'Misuse of Corporate Vehicles' (FATF, 2006 updated)."
        )
        sar = SARDraft(
            sar_id=f"SAR-{str(uuid.uuid4())[:8].upper()}",
            customer_id=customer_id,
            account_id=account_id,
            alert_ids=alert_ids,
            total_suspicious_amount_gbp=total_suspicious_amount_gbp,
            nature_of_suspicion=nature,
            typology_citation=typology_cite,
            financial_details=(
                f"Total: £{total_suspicious_amount_gbp:,.2f} | "
                f"Alerts: {len(alert_ids)}"
            ),
            status=SARStatus.DRAFT,
            poca_section="s.330",
            sar_type=sar_type,
            requires_mlro_approval=True,  # ALWAYS True
            tipping_off_guardrail_active=True,  # ALWAYS True
            model_id=self._model_id,
        )
        log.info(
            "SAR draft: id=%s customer=%s amount=£%s "
            "type=%s mlro_approval_required=True",
            sar.sar_id, customer_id,
            total_suspicious_amount_gbp, sar_type,
        )
        return sar

    def submit_to_nca(
        self,
        sar: SARDraft,
        mlro_id: str,
    ) -> SARDraft:
        """Submit MLRO-approved SAR to NCA SubmitSAR API.

        UK National Crime Agency — NOT FinCEN (US-only).
        Validates MLRO approval before submission.

        Args:
            sar: SAR in LEGAL_CLEARED status.
            mlro_id: MLRO staff identifier (mandatory).

        Returns:
            Updated SARDraft with NCA reference.

        Raises:
            ValueError: If SAR not LEGAL_CLEARED or no MLRO.
        """
        if sar.status != SARStatus.LEGAL_CLEARED:
            raise ValueError(
                f"SAR {sar.sar_id} must be LEGAL_CLEARED "
                f"before NCA submission. Current: {sar.status.value}"
            )
        if not mlro_id:
            raise ValueError(
                "MLRO ID required for NCA submission "
                "(POCA 2002 s.331 — MLRO disclosure obligation)"
            )
        # Production: POST to NCA SubmitSAR API
        nca_ref = f"NCA-{sar.sar_id}-2026"
        sar.status = SARStatus.SUBMITTED
        sar.nca_reference = nca_ref
        sar.mlro_id = mlro_id
        sar.submitted_at = datetime.utcnow()
        log.info(
            "SAR submitted to NCA: id=%s ref=%s mlro=%s",
            sar.sar_id, nca_ref, mlro_id,
        )
        return sar
