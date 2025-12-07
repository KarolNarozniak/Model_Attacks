import argparse
import os
import re
from typing import Optional, Tuple

from infer_bert_tiny_qa import answer_question


REFUSAL_MESSAGE = (
    "I can't disclose private information or help de-anonymize individuals."
)


SENSITIVE_PATTERNS = [
    # Prompt-injection / jailbreak intents
    re.compile(r"ignore (all|previous) instructions", re.IGNORECASE),
    re.compile(r"pretend (to|that)", re.IGNORECASE),
    re.compile(r"jailbreak|bypass|override policy", re.IGNORECASE),

    # Membership inference / dataset inclusion
    re.compile(r"(czy|is) .* (jest|was) .* (w treningu|w zbiorze|in the training|in the dataset)", re.IGNORECASE),
    re.compile(r"(należy|nalezy|belongs) .* (do treningu|to training)", re.IGNORECASE),

    # Attribute inference (identity linkage)
    re.compile(r"(na którym roku|na ktorym roku|which year).* (stud(uje|entem)|study)", re.IGNORECASE),
    re.compile(r"(jaki|what) (numer|number).*(albumu|indeksu|index)", re.IGNORECASE),
    re.compile(r"(data urodzenia|birth\s*date|wiek|age)", re.IGNORECASE),

    # Secrets and credentials
    re.compile(r"hasło|password|secret|api\s*key|token|private\s*key", re.IGNORECASE),

    # PII indicators (emails, phones, national IDs)
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),  # email
    re.compile(r"\b(\+?48)?[-\s.]?\d{3}[-\s.]?\d{3}[-\s.]?\d{3}\b"),   # PL phone
    re.compile(r"\b\d{11}\b"),  # PESEL-likes (very rough)
    re.compile(r"(adres|address|ul\.|ulica)\s+.+", re.IGNORECASE),
]


def is_sensitive_query(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return "empty query"
    for pat in SENSITIVE_PATTERNS:
        if pat.search(t):
            return f"blocked by pattern: {pat.pattern}"
    # Heuristic: questions explicitly about a named individual + status
    if re.search(r"\b(Czy|Is)\b.+\b(studentem|student|member)\b", t, re.IGNORECASE):
        return "membership-style query"
    return None


def gateway_answer(
    question: str,
    logit_noise: str = "none",
    noise_scale: float = 0.0,
    seed: int = 42,
) -> Tuple[str, float, bool, str]:
    """
    Returns (message_or_answer, confidence, allowed, reason).
    If not allowed, confidence=0.0 and message is a refusal.
    """
    reason = is_sensitive_query(question)
    if reason is not None:
        return REFUSAL_MESSAGE, 0.0, False, reason

    # Allow inference with optional noise for privacy hardening
    ans, conf = answer_question(
        question,
        logit_noise=logit_noise,
        noise_scale=noise_scale,
        seed=seed,
    )
    return ans, conf, True, "allowed"


def main():
    parser = argparse.ArgumentParser(description="Safeguard gateway for model access")
    parser.add_argument("--q", dest="question", required=True, help="User question")
    parser.add_argument("--logit_noise", choices=["none", "gaussian", "laplace"], default=os.getenv("GATEWAY_LOGIT_NOISE", "none"))
    parser.add_argument("--noise_scale", type=float, default=float(os.getenv("GATEWAY_NOISE_SCALE", "0.0")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("GATEWAY_SEED", "42")))
    args = parser.parse_args()

    msg, conf, allowed, reason = gateway_answer(
        args.question,
        logit_noise=args.logit_noise,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    if allowed:
        print(f"Answer: {msg}")
        print(f"Confidence: {conf:.3f}")
    else:
        print(msg)
        print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
