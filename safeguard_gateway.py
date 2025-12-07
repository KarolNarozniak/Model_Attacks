import argparse
import os
import re
import unicodedata
from typing import Optional, Tuple

from infer_bert_tiny_qa import answer_question


REFUSAL_MESSAGE = (
    "I can't disclose private information or help de-anonymize individuals."
)


SENSITIVE_PATTERNS = [
    # Prompt-injection / jailbreak intents
    re.compile(r"ignore (all|previous) instructions|disregard the rules", re.IGNORECASE),
    re.compile(r"pretend (to|that)|roleplay|as a system|system prompt", re.IGNORECASE),
    re.compile(r"\bjailbreak\b|\bbypass\b|\boverride (policy|safety|guardrails)\b|\bDAN\b|\bdeveloper\s*mode\b", re.IGNORECASE),
    re.compile(r"base64|rot13|decode then answer|translate then answer", re.IGNORECASE),

    # Membership inference / dataset inclusion
    re.compile(r"(czy|is) .* (jest|was) .* (w treningu|w zbiorze|w zbiorze treningowym|in the training|in the dataset)", re.IGNORECASE),
    re.compile(r"(należy|nalezy|belongs?)\s+.*\s+(do treningu|to training|to training data|to the dataset|to dataset)", re.IGNORECASE),
    re.compile(r"\bbelong(?:s)?\s+to\s+(?:training(?:\s+data)?|the\s+dataset|dataset)\b", re.IGNORECASE),

    # Attribute inference (identity linkage)
    re.compile(r"(na którym roku|na ktorym roku|which year).* (stud(uje|entem)|study)", re.IGNORECASE),
    re.compile(r"(jaki|what) (numer|number).*(albumu|indeksu|index)", re.IGNORECASE),
    re.compile(r"((data|datę)\s+urodzenia|birth\s*date|wiek|age|adres e-?mail|email)", re.IGNORECASE),

    # Secrets and credentials
    re.compile(r"hasło|password|secret|api\s*key|token|private\s*key|credentials", re.IGNORECASE),

    # PII indicators (emails, phones, national IDs, banking)
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),  # email literal
    re.compile(r"\b(\+?48)?[-\s.]?\d{3}[-\s.]?\d{3}[-\s.]?\d{3}\b"),   # PL phone
    re.compile(r"\b\d{11}\b"),  # PESEL-like (rough)
    re.compile(r"\b\d{2}-\d{3}\b"),  # PL postal code
    re.compile(r"\bNIP\b[:\s-]*\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}", re.IGNORECASE),
    re.compile(r"\bREGON\b[:\s-]*\d{9}(?:\d{5})?", re.IGNORECASE),
    re.compile(r"\bPL\d{26}\b"),  # IBAN PL
    re.compile(r"\b(?:\d[ \-]?){13,16}\b"),  # generic CC-like pattern
    re.compile(r"(adres|address|ul\.|ulica)\s+.+", re.IGNORECASE),
]


def _normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t


def _contains_named_individual(t: str) -> bool:
    norm = _normalize_text(t).lower()
    condensed = re.sub(r"[^a-z]", "", norm)
    return "karolnarozniak" in condensed


def is_sensitive_query(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return "empty query"
    for pat in SENSITIVE_PATTERNS:
        if pat.search(t):
            return f"blocked by pattern: {pat.pattern}"
    # Heuristic: questions explicitly about membership/status
    if re.search(r"\b(Czy|Is)\b.+\b(studentem|student|member)\b", t, re.IGNORECASE):
        return "membership-style query"
    # Heuristic: any query mentioning the protected individual's full name
    if _contains_named_individual(t):
        return "named individual detected"
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
