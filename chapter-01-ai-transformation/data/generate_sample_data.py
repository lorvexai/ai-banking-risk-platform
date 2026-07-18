"""
data/generate_sample_data.py
AWB AI Customer Service Platform — Sample Data Generator

Generates sample customer messages for testing, development, and demo.
All data is fictional — no real customer PII.

Run: python data/generate_sample_data.py
Output: data/sample_messages.json

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance in Banking
"""

import json
import random
import uuid
from datetime import datetime, timezone

# Sample messages by intent (representative of AWB customer base)
SAMPLE_MESSAGES: dict[str, list[str]] = {
    "balance_enquiry": [
        "What is my current account balance?",
        "Can you tell me how much is in my savings account?",
        "What's my available balance please?",
        "How much money do I have in my account?",
        "Check my balance",
        "I need to know my current balance before making a payment.",
    ],
    "product_enquiry": [
        "Tell me about your ISA accounts.",
        "What SME loans do you offer?",
        "I'm interested in a business loan — what are the terms?",
        "Do you have any fixed-rate savings products?",
        "What's the difference between your current accounts?",
        "Can you explain how your Cash ISA works?",
    ],
    "rate_enquiry": [
        "What interest rate do you pay on savings?",
        "What's the current mortgage rate?",
        "How much interest will I earn on £10,000 in your ISA?",
        "What's the APR on your business loans?",
        "Are your savings rates competitive right now?",
    ],
    "payment_support": [
        "I need help setting up a standing order.",
        "How do I cancel a direct debit?",
        "My payment didn't go through — what do I do?",
        "Can I make an international transfer?",
        "I need to change my payment date.",
    ],
    "complaint": [
        "I want to make a formal complaint about the service.",
        "This is absolutely unacceptable — I've been waiting 3 weeks.",
        "I am very unhappy with how my account was handled.",
        "I need to speak to someone about a serious issue with my account.",
        "Your staff gave me completely wrong information about my mortgage.",
    ],
    "account_change": [
        "I want to change my address on my account.",
        "Can you update my phone number?",
        "I need to change my account sort code.",
        "How do I update my bank account details?",
        "I want to switch to a different account type.",
    ],
    "out_of_scope": [
        "What's the weather like today?",
        "Tell me a joke.",
        "Who won the Premier League last season?",
        "What is the capital of Australia?",
        "Can you write me a poem?",
    ],
}

CHANNELS = ["web", "app", "ivr"]
SEGMENTS = ["retail", "sme", "private"]


def generate_sample_messages(
    n_per_intent: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    Generate n_per_intent sample messages per intent category.

    Args:
        n_per_intent: Number of messages per intent to include.
        seed:         Random seed for reproducibility.

    Returns:
        List of message dicts with session_id, customer_id, message, expected_intent.
    """
    random.seed(seed)
    samples = []

    for intent, messages in SAMPLE_MESSAGES.items():
        selected = random.sample(messages, min(n_per_intent, len(messages)))
        for msg in selected:
            samples.append({
                "session_id": str(uuid.uuid4()),
                "customer_id": f"CUST-{random.randint(10000, 99999)}",
                "message": msg,
                "expected_intent": intent,
                "channel": random.choice(CHANNELS),
                "customer_segment": random.choice(SEGMENTS),
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            })

    random.shuffle(samples)
    return samples


def main():
    samples = generate_sample_messages(n_per_intent=3)
    output_path = "data/sample_messages.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} sample messages → {output_path}")

    # Print summary by intent
    from collections import Counter
    counts = Counter(s["expected_intent"] for s in samples)
    print("\nBreakdown by intent:")
    for intent, count in sorted(counts.items()):
        print(f"  {intent:<25} {count:>3} messages")


if __name__ == "__main__":
    main()
