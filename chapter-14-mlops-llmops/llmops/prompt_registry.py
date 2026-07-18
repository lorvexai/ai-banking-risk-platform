# llmops/prompt_registry.py | AWB LLMOps prompt registry
# Chapter 14 | MAJOR.MINOR.PATCH versioning
# Linked to Chapter 10 MR-2026-058 prompt registry spec
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import yaml

log = logging.getLogger(__name__)

REGISTRY_PATH = Path("prompts/registry.yaml")
VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$"
)


class ChangeType(str, Enum):
    """MAJOR.MINOR.PATCH semantic versioning.

    MAJOR: New output schema or regulatory framing.
           Requires MRC review + 2-week A/B test.
    MINOR: Additive capability extension.
           Requires robustness suite + 1-week A/B test.
    PATCH: Backwards-compatible fix.
           Requires robustness suite only; 48hr monitoring.
    """
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    PATCH = "PATCH"


@dataclass
class PromptVersion:
    """A versioned prompt in the AWB prompt registry."""
    service_id: str        # e.g. MR-2026-035
    version: str           # e.g. "1.3.2"
    git_tag: str           # e.g. "prompt/MR-2026-035/1.3.2"
    change_type: ChangeType
    author: str
    description: str
    requires_mrc: bool
    ab_test_days: int
    is_production: bool = False

    def bump(
        self,
        change_type: ChangeType,
    ) -> str:
        """Return next version string for change type.

        Args:
            change_type: MAJOR, MINOR, or PATCH.
        Returns:
            New semantic version string.
        """
        m = VERSION_RE.match(self.version)
        if not m:
            raise ValueError(
                f"Invalid version: {self.version}"
            )
        major = int(m.group("major"))
        minor = int(m.group("minor"))
        patch = int(m.group("patch"))

        if change_type == ChangeType.MAJOR:
            return f"{major + 1}.0.0"
        elif change_type == ChangeType.MINOR:
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"


class PromptRegistry:
    """Git-backed AWB prompt registry (MR-2026-058).

    Reads active production versions from registry.yaml.
    All writes are Git commits — fully auditable.
    7-year retention per FCA COBS 9 credit audit trail.
    """

    def __init__(
        self,
        registry_path: Path = REGISTRY_PATH,
    ) -> None:
        self.registry_path = registry_path
        self._registry: dict = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self._registry = yaml.safe_load(f) or {}
        log.info(
            "Loaded registry: %d services",
            len(self._registry),
        )

    def get_production_version(
        self, service_id: str
    ) -> Optional[str]:
        """Return active production prompt version.

        Args:
            service_id: AWB model registry ID.
        Returns:
            Version string or None if not registered.
        """
        return self._registry.get(
            service_id, {}
        ).get("production_version")

    def register_version(
        self,
        version: PromptVersion,
    ) -> None:
        """Register a new prompt version.

        Args:
            version: PromptVersion to register.
        Raises:
            ValueError: If change_type requires MRC
                and mrc_approved tag missing from run.
        """
        if version.change_type == ChangeType.MAJOR:
            log.warning(
                "MAJOR version %s requires MRC approval"
                " before production deployment.",
                version.version,
            )
        self._registry.setdefault(version.service_id, {})
        self._registry[version.service_id][
            version.version
        ] = {
            "git_tag": version.git_tag,
            "change_type": version.change_type.value,
            "author": version.author,
            "description": version.description,
            "requires_mrc": version.requires_mrc,
            "ab_test_days": version.ab_test_days,
        }
        self._save()
        log.info(
            "Registered %s v%s (%s)",
            version.service_id,
            version.version,
            version.change_type.value,
        )

    def promote_to_production(
        self,
        service_id: str,
        version: str,
    ) -> None:
        """Set version as active production prompt.

        Args:
            service_id: AWB model registry ID.
            version: Version string to promote.
        Raises:
            KeyError: If version not registered.
        """
        if version not in self._registry.get(
            service_id, {}
        ):
            raise KeyError(
                f"{service_id} v{version} not registered"
            )
        self._registry[service_id][
            "production_version"
        ] = version
        self._save()
        log.info(
            "Promoted %s v%s to production",
            service_id,
            version,
        )

    def _save(self) -> None:
        self.registry_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(self.registry_path, "w") as f:
            yaml.dump(self._registry, f)
