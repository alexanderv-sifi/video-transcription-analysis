"""
Dictionary-based transcript correction system.

This module provides pattern-based text correction for transcripts, fixing
common transcription errors and standardizing terminology. It's the second
tier in the correction pipeline (after vocabulary hints, before LLM).

Architecture:
- DictionaryCorrector: Main interface for applying corrections
- YAML-based correction rules for easy editing
- Regex support for flexible pattern matching
- Statistics tracking for monitoring effectiveness

Example:
    >>> from pathlib import Path
    >>> corrector = DictionaryCorrector(Path("corrections.yaml"))
    >>> text = "We use as was for the Ed wiz platform"
    >>> corrected, stats = corrector.correct(text)
    >>> print(corrected)
    "We use AdsWizz for the AdsWizz platform"
    >>> print(stats)
    {"corrections_applied": 2, "patterns_matched": ["as was", "Ed wiz"]}
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

import yaml

logger = logging.getLogger(__name__)


class CorrectorError(Exception):
    """Base exception for corrector errors."""
    pass


@dataclass
class CorrectionRule:
    """
    A single correction rule with pattern and replacement.

    Attributes:
        pattern: Text pattern to match (string or regex)
        replacement: Text to replace with
        case_sensitive: Whether matching is case-sensitive
        is_regex: Whether pattern is a regex (auto-detected)
        compiled_pattern: Compiled regex pattern (cached)
    """

    pattern: str
    replacement: str
    case_sensitive: bool = False
    is_regex: bool = False
    compiled_pattern: Optional[Pattern] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Compile the pattern for efficient matching."""
        # Auto-detect regex patterns
        regex_chars = r"[.*+?^${}()|[\]\\]"
        self.is_regex = bool(re.search(regex_chars, self.pattern))

        # Compile pattern
        if self.is_regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                self.compiled_pattern = re.compile(self.pattern, flags)
            except re.error as e:
                raise CorrectorError(f"Invalid regex pattern '{self.pattern}': {e}") from e
        else:
            # For literal strings, escape and create pattern
            escaped_pattern = re.escape(self.pattern)
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self.compiled_pattern = re.compile(
                r"\b" + escaped_pattern + r"\b",  # Word boundaries for exact matches
                flags
            )

    def apply(self, text: str) -> Tuple[str, int]:
        """
        Apply this correction rule to text.

        Args:
            text: Text to correct

        Returns:
            Tuple of (corrected_text, number_of_replacements)
        """
        if not self.compiled_pattern:
            return text, 0

        # Count matches before replacement
        matches = self.compiled_pattern.findall(text)
        count = len(matches)

        if count > 0:
            text = self.compiled_pattern.sub(self.replacement, text)
            logger.debug(
                f"Applied correction '{self.pattern}' → '{self.replacement}': {count} times"
            )

        return text, count


@dataclass
class CorrectionStats:
    """Statistics about corrections applied."""

    total_corrections: int = 0
    patterns_matched: List[str] = field(default_factory=list)
    pattern_counts: Dict[str, int] = field(default_factory=dict)

    def add(self, pattern: str, count: int) -> None:
        """Record a correction application."""
        if count > 0:
            self.total_corrections += count
            if pattern not in self.patterns_matched:
                self.patterns_matched.append(pattern)
            self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + count

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_corrections": self.total_corrections,
            "patterns_matched": self.patterns_matched,
            "pattern_counts": self.pattern_counts,
        }


class DictionaryCorrector:
    """
    Apply dictionary-based corrections to transcripts.

    This class loads correction rules from a YAML file and applies them
    to transcript text. It supports both literal string matching and
    regex patterns, with optional case sensitivity.

    Attributes:
        corrections_path: Path to YAML corrections file
        rules: List of correction rules

    Design decisions:
        - YAML storage for easy manual editing
        - Compiled regex patterns for performance
        - Word boundary matching to avoid partial word replacements
        - Statistics tracking for monitoring effectiveness
        - Preserves speaker labels (doesn't correct SPEAKER_XX markers)
    """

    def __init__(self, corrections_path: Path):
        """
        Initialize dictionary corrector.

        Args:
            corrections_path: Path to YAML corrections file

        Raises:
            CorrectorError: If corrections file is corrupted
        """
        self.corrections_path = corrections_path
        self.rules: List[CorrectionRule] = []

        # Load existing corrections
        self._load_corrections()

        logger.info(
            f"Initialized DictionaryCorrector with {len(self.rules)} rules"
        )

    def add_rule(
        self,
        pattern: str,
        replacement: str,
        case_sensitive: bool = False,
        save: bool = True,
    ) -> None:
        """
        Add a new correction rule.

        Args:
            pattern: Text pattern to match
            replacement: Text to replace with
            case_sensitive: Whether matching is case-sensitive
            save: If True, save corrections to file immediately

        Raises:
            ValueError: If pattern or replacement is empty
            CorrectorError: If pattern is invalid regex
        """
        if not pattern or not pattern.strip():
            raise ValueError("Pattern cannot be empty")
        if not replacement or not replacement.strip():
            raise ValueError("Replacement cannot be empty")

        # Check for duplicates
        for rule in self.rules:
            if (
                rule.pattern == pattern
                and rule.replacement == replacement
                and rule.case_sensitive == case_sensitive
            ):
                logger.debug(f"Rule already exists: '{pattern}' → '{replacement}'")
                return

        # Create and add rule
        rule = CorrectionRule(
            pattern=pattern.strip(),
            replacement=replacement.strip(),
            case_sensitive=case_sensitive,
        )
        self.rules.append(rule)

        logger.info(f"Added correction rule: '{pattern}' → '{replacement}'")

        if save:
            self._save_corrections()

    def remove_rule(self, pattern: str, save: bool = True) -> bool:
        """
        Remove a correction rule by pattern.

        Args:
            pattern: Pattern of rule to remove
            save: If True, save corrections to file immediately

        Returns:
            True if rule was removed, False if not found
        """
        initial_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.pattern != pattern]

        removed = len(self.rules) < initial_count

        if removed:
            logger.info(f"Removed correction rule: '{pattern}'")
            if save:
                self._save_corrections()

        return removed

    def correct(self, text: str) -> Tuple[str, CorrectionStats]:
        """
        Apply all correction rules to text.

        Args:
            text: Text to correct (can include speaker labels)

        Returns:
            Tuple of (corrected_text, statistics)

        Note:
            Speaker labels (SPEAKER_00:, etc.) are preserved and not corrected.
        """
        if not self.rules:
            return text, CorrectionStats()

        stats = CorrectionStats()
        corrected = text

        # Split by lines to preserve speaker labels
        lines = corrected.split("\n")
        corrected_lines = []

        for line in lines:
            # Check if line has speaker label
            speaker_label = ""
            content = line

            if ":" in line and line.startswith("SPEAKER_"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    speaker_label = parts[0] + ": "
                    content = parts[1]

            # Apply corrections to content only
            for rule in self.rules:
                content, count = rule.apply(content)
                if count > 0:
                    stats.add(rule.pattern, count)

            # Reconstruct line
            corrected_lines.append(speaker_label + content)

        corrected = "\n".join(corrected_lines)

        if stats.total_corrections > 0:
            logger.info(
                f"Applied {stats.total_corrections} corrections using "
                f"{len(stats.patterns_matched)} patterns"
            )

        return corrected, stats

    def get_rules(self) -> List[CorrectionRule]:
        """Get all correction rules."""
        return list(self.rules)

    def clear_rules(self, save: bool = True) -> None:
        """
        Clear all correction rules.

        Args:
            save: If True, save corrections to file immediately
        """
        self.rules = []
        logger.info("Cleared all correction rules")

        if save:
            self._save_corrections()

    def _load_corrections(self) -> None:
        """Load corrections from YAML file."""
        if not self.corrections_path.exists():
            logger.info(f"No existing corrections found at {self.corrections_path}")
            return

        try:
            with open(self.corrections_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("Corrections file is empty")
                return

            # Validate format
            if "corrections" not in data:
                raise ValueError("Invalid corrections format: missing 'corrections' key")

            corrections_list = data["corrections"]
            if not isinstance(corrections_list, list):
                raise ValueError("'corrections' must be a list")

            # Load rules
            for item in corrections_list:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping invalid correction item: {item}")
                    continue

                if "pattern" not in item or "replacement" not in item:
                    logger.warning(f"Skipping correction without pattern/replacement: {item}")
                    continue

                rule = CorrectionRule(
                    pattern=item["pattern"],
                    replacement=item["replacement"],
                    case_sensitive=item.get("case_sensitive", False),
                )
                self.rules.append(rule)

            logger.info(f"Loaded {len(self.rules)} correction rules from {self.corrections_path}")

        except yaml.YAMLError as e:
            raise CorrectorError(
                f"Corrupted corrections file at {self.corrections_path}: {e}"
            ) from e
        except Exception as e:
            raise CorrectorError(
                f"Failed to load corrections from {self.corrections_path}: {e}"
            ) from e

    def _save_corrections(self) -> None:
        """Save corrections to YAML file."""
        try:
            # Ensure directory exists
            self.corrections_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data
            corrections_list = []
            for rule in self.rules:
                item = {
                    "pattern": rule.pattern,
                    "replacement": rule.replacement,
                }
                if rule.case_sensitive:
                    item["case_sensitive"] = True
                corrections_list.append(item)

            data = {"corrections": corrections_list}

            # Write atomically
            temp_path = self.corrections_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                yaml.safe_dump(
                    data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
                )

            temp_path.replace(self.corrections_path)
            logger.debug(f"Saved corrections to {self.corrections_path}")

        except Exception as e:
            raise CorrectorError(
                f"Failed to save corrections to {self.corrections_path}: {e}"
            ) from e
