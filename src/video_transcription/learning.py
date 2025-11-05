"""
Correction learning from manual transcript edits.

This module analyzes differences between original and manually-edited transcripts
to extract correction patterns. It suggests dictionary rules and vocabulary terms
based on observed changes.

Architecture:
- CorrectionLearner: Main interface for learning from edits
- Word-level diffing with difflib for accurate alignment
- Pattern extraction with frequency analysis
- Noise filtering for high-quality suggestions

Example:
    >>> learner = CorrectionLearner(min_occurrences=2)
    >>> original = "We use as was for Ed wiz platform"
    >>> edited = "We use AdsWizz for AdsWizz platform"
    >>> result = learner.learn_from_diff(original, edited)
    >>> suggestions = learner.suggest_rules(result)
    >>> for suggestion in suggestions:
    ...     print(f"{suggestion.type}: {suggestion.pattern} → {suggestion.replacement}")
"""

import difflib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class LearningError(Exception):
    """Base exception for learning errors."""
    pass


@dataclass
class Change:
    """A single detected change from original to edited."""

    original: str
    edited: str
    context_before: str = ""
    context_after: str = ""
    position: int = 0

    def __hash__(self):
        """Make Change hashable for use in sets/dicts."""
        return hash((self.original, self.edited))


@dataclass
class Suggestion:
    """A suggested correction rule or vocabulary term."""

    type: str  # "dictionary" or "vocabulary"
    pattern: str
    replacement: Optional[str] = None
    occurrences: int = 0
    confidence: float = 0.0
    contexts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "pattern": self.pattern,
            "replacement": self.replacement,
            "occurrences": self.occurrences,
            "confidence": self.confidence,
            "contexts": self.contexts[:3],  # Limit to 3 examples
        }


@dataclass
class LearningResult:
    """Result of learning from diff."""

    changes: List[Change]
    total_words: int
    words_changed: int
    change_rate: float
    pattern_frequency: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_changes": len(self.changes),
            "total_words": self.total_words,
            "words_changed": self.words_changed,
            "change_rate": self.change_rate,
            "unique_patterns": len(self.pattern_frequency),
        }


class CorrectionLearner:
    """
    Learn correction patterns from manual transcript edits.

    This class analyzes differences between original and edited transcripts
    to extract useful correction patterns. It uses word-level diffing to
    accurately identify changes and filters noise to provide high-quality
    suggestions for dictionary rules and vocabulary terms.

    Attributes:
        min_occurrences: Minimum times a pattern must appear to suggest
        min_confidence: Minimum confidence threshold for suggestions
        context_window: Number of words before/after for context

    Design decisions:
        - Word-level diffing (better than line or char level)
        - Frequency-based confidence scoring
        - Context awareness for disambiguation
        - Noise filtering (whitespace, capitalization, speaker labels)
        - Separate suggestions for dictionary vs vocabulary
    """

    def __init__(
        self,
        min_occurrences: int = 2,
        min_confidence: float = 0.7,
        context_window: int = 2,
    ):
        """
        Initialize correction learner.

        Args:
            min_occurrences: Minimum times pattern must appear (default: 2)
            min_confidence: Minimum confidence for suggestions (default: 0.7)
            context_window: Words before/after for context (default: 2)

        Raises:
            ValueError: If parameters are invalid
        """
        if min_occurrences < 1:
            raise ValueError("min_occurrences must be at least 1")

        if min_confidence < 0 or min_confidence > 1:
            raise ValueError("min_confidence must be between 0 and 1")

        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence
        self.context_window = context_window

        logger.info(
            f"Initialized CorrectionLearner: min_occurrences={min_occurrences}, "
            f"min_confidence={min_confidence}"
        )

    def learn_from_diff(self, original: str, edited: str) -> LearningResult:
        """
        Learn correction patterns from diff between original and edited text.

        Args:
            original: Original transcript text
            edited: Manually edited transcript text

        Returns:
            LearningResult with detected changes and statistics

        Raises:
            LearningError: If diff analysis fails
        """
        if not original or not edited:
            raise LearningError("Both original and edited text must be non-empty")

        try:
            # Tokenize into words (preserving speaker labels)
            original_words = self._tokenize(original)
            edited_words = self._tokenize(edited)

            logger.info(
                f"Analyzing diff: {len(original_words)} → {len(edited_words)} words"
            )

            # Extract changes using difflib
            changes = self._extract_changes(original_words, edited_words)

            # Filter noise
            changes = self._filter_noise(changes)

            # Count pattern frequency
            pattern_freq = Counter((c.original, c.edited) for c in changes)

            # Calculate statistics
            words_changed = len(set(c.position for c in changes))
            change_rate = words_changed / len(original_words) if original_words else 0

            result = LearningResult(
                changes=changes,
                total_words=len(original_words),
                words_changed=words_changed,
                change_rate=change_rate,
                pattern_frequency=dict(pattern_freq),
            )

            logger.info(
                f"✓ Found {len(changes)} changes ({change_rate:.1%} of words), "
                f"{len(pattern_freq)} unique patterns"
            )

            return result

        except Exception as e:
            raise LearningError(f"Failed to learn from diff: {e}") from e

    def suggest_rules(self, result: LearningResult) -> List[Suggestion]:
        """
        Generate suggestions for dictionary rules and vocabulary terms.

        Args:
            result: Learning result from learn_from_diff()

        Returns:
            List of suggestions sorted by confidence

        Note:
            - Dictionary suggestions: Consistent replacements (pattern → replacement)
            - Vocabulary suggestions: New terms that should be recognized
        """
        suggestions = []

        # Analyze each unique pattern
        for (original, edited), count in result.pattern_frequency.items():
            # Skip if below minimum occurrences
            if count < self.min_occurrences:
                continue

            # Calculate confidence based on frequency and consistency
            confidence = self._calculate_confidence(
                original, edited, count, result.total_words
            )

            # Skip if below confidence threshold
            if confidence < self.min_confidence:
                continue

            # Collect contexts
            contexts = [
                self._format_context(c)
                for c in result.changes
                if c.original == original and c.edited == edited
            ][:3]  # Limit to 3 examples

            # Determine suggestion type
            if self._is_vocabulary_candidate(original, edited):
                # Vocabulary term (for initial_prompt)
                suggestions.append(
                    Suggestion(
                        type="vocabulary",
                        pattern=edited,  # The correct term
                        replacement=None,
                        occurrences=count,
                        confidence=confidence,
                        contexts=contexts,
                    )
                )

            # Always suggest dictionary rule for consistent replacements
            suggestions.append(
                Suggestion(
                    type="dictionary",
                    pattern=original,
                    replacement=edited,
                    occurrences=count,
                    confidence=confidence,
                    contexts=contexts,
                )
            )

        # Sort by confidence (high to low)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(f"Generated {len(suggestions)} suggestions")
        return suggestions

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words while preserving speaker labels.

        Args:
            text: Text to tokenize

        Returns:
            List of words
        """
        # Split into words, preserving speaker labels as single tokens
        words = []
        for line in text.split("\n"):
            # Check if line has speaker label
            if ":" in line and line.strip().startswith("SPEAKER_"):
                # Split speaker label from content
                parts = line.split(":", 1)
                if len(parts) == 2:
                    words.append(parts[0] + ":")  # Keep speaker label as token
                    line = parts[1]

            # Tokenize remaining content
            line_words = re.findall(r"\S+", line)
            words.extend(line_words)

        return words

    def _extract_changes(
        self, original_words: List[str], edited_words: List[str]
    ) -> List[Change]:
        """
        Extract changes between word lists using difflib.

        Args:
            original_words: Original word list
            edited_words: Edited word list

        Returns:
            List of detected changes
        """
        changes = []
        matcher = difflib.SequenceMatcher(None, original_words, edited_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                # Word(s) were replaced
                original_phrase = " ".join(original_words[i1:i2])
                edited_phrase = " ".join(edited_words[j1:j2])

                # Get context
                context_before = " ".join(
                    original_words[max(0, i1 - self.context_window) : i1]
                )
                context_after = " ".join(
                    original_words[i2 : i2 + self.context_window]
                )

                changes.append(
                    Change(
                        original=original_phrase,
                        edited=edited_phrase,
                        context_before=context_before,
                        context_after=context_after,
                        position=i1,
                    )
                )

        return changes

    def _filter_noise(self, changes: List[Change]) -> List[Change]:
        """
        Filter out noisy changes (whitespace, capitalization, etc.).

        Args:
            changes: List of changes

        Returns:
            Filtered list of changes
        """
        filtered = []

        for change in changes:
            # Skip if only whitespace/punctuation differs
            original_clean = re.sub(r"[\s\.,!?;:]", "", change.original.lower())
            edited_clean = re.sub(r"[\s\.,!?;:]", "", change.edited.lower())

            if original_clean == edited_clean:
                continue

            # Skip very short changes (likely typos)
            if len(change.original) < 2 and len(change.edited) < 2:
                continue

            # Skip speaker label changes (probably intentional)
            if change.original.startswith("SPEAKER_") or change.edited.startswith(
                "SPEAKER_"
            ):
                continue

            filtered.append(change)

        return filtered

    def _calculate_confidence(
        self, original: str, edited: str, occurrences: int, total_words: int
    ) -> float:
        """
        Calculate confidence score for a pattern.

        Confidence is based on:
        - Frequency (how often the pattern appears)
        - Consistency (always corrected the same way)
        - Distinctiveness (not just capitalization)

        Args:
            original: Original text
            edited: Edited text
            occurrences: Number of times pattern appears
            total_words: Total words in transcript

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from frequency (logarithmic scale)
        frequency_score = min(1.0, occurrences / 5.0)

        # Boost for clear corrections (very different strings)
        similarity = difflib.SequenceMatcher(None, original.lower(), edited.lower()).ratio()
        distinctiveness_score = 1.0 - similarity

        # Boost for consistent length changes (not just typos)
        length_ratio = abs(len(edited) - len(original)) / max(len(original), 1)
        consistency_score = min(1.0, length_ratio)

        # Combine scores (weighted average)
        confidence = (
            0.5 * frequency_score + 0.3 * distinctiveness_score + 0.2 * consistency_score
        )

        return min(1.0, confidence)

    def _is_vocabulary_candidate(self, original: str, edited: str) -> bool:
        """
        Determine if change suggests a vocabulary term.

        Vocabulary candidates are:
        - Multi-word terms → single terms (e.g., "as was" → "AdsWizz")
        - Capitalized proper nouns
        - Technical terms with special formatting

        Args:
            original: Original text
            edited: Edited text

        Returns:
            True if likely a vocabulary term
        """
        # Multi-word → single-word (compound correction)
        if " " in original and " " not in edited:
            return True

        # Starts with capital (proper noun)
        if edited and edited[0].isupper():
            return True

        # Has special formatting (camelCase, acronyms, etc.)
        if re.search(r"[A-Z]{2,}", edited):  # ACRONYM
            return True

        if re.search(r"[a-z][A-Z]", edited):  # camelCase
            return True

        return False

    def _format_context(self, change: Change) -> str:
        """
        Format change context for display.

        Args:
            change: Change to format

        Returns:
            Formatted context string
        """
        before = change.context_before[-20:] if change.context_before else ""
        after = change.context_after[:20] if change.context_after else ""

        return f"{before} [{change.original} → {change.edited}] {after}".strip()
