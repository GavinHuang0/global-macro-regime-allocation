"""Shared domain types for the rewritten pipeline."""

from enum import Enum


class MacroRegime(str, Enum):
    INFLATIONARY_BOOM = "inflationary_boom"
    DEFLATIONARY_BOOM = "deflationary_boom"
    INFLATIONARY_BUST = "inflationary_bust"
    DEFLATIONARY_BUST = "deflationary_bust"
