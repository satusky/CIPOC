"""LLM agent prompts."""

from .extractor import EXTRACTOR_SYSTEM_PROMPT, EXTRACT_VALUES_PROMPT, extractor_user_prompt
from .note_scanner import (
    NOTE_SCANNER_SYSTEM_PROMPT,
    CANCER_IN_NOTE_PROMPT,
    NOTE_SUMMARY_PROMPT,
    CANCER_MENTIONS_PROMPT,
)

__all__ = [
    "EXTRACTOR_SYSTEM_PROMPT",
    "EXTRACT_VALUES_PROMPT",
    "extractor_user_prompt",
    "NOTE_SCANNER_SYSTEM_PROMPT",
    "CANCER_IN_NOTE_PROMPT",
    "NOTE_SUMMARY_PROMPT",
    "CANCER_MENTIONS_PROMPT",
]
