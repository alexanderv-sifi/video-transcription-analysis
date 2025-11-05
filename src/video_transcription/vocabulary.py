"""
Custom vocabulary management for improved transcription accuracy.

This module provides vocabulary management to help Whisper recognize domain-specific
terms, company names, technical jargon, and speaker names. The vocabulary is used to
build an optimized initial_prompt that guides Whisper's transcription.

Architecture:
- VocabularyManager: Main interface for vocabulary operations
- YAML-based storage for human-readable editing
- Automatic integration with speaker database
- Optimized prompt building (avoids token limits)

Example:
    >>> from pathlib import Path
    >>> vocab = VocabularyManager(Path("vocabulary.yaml"))
    >>> vocab.add_term("AdsWizz", category="companies")
    >>> vocab.add_term("SFTP", category="technical_terms")
    >>> prompt = vocab.build_initial_prompt()
    >>> # Use prompt with Whisper transcription
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class VocabularyError(Exception):
    """Base exception for vocabulary errors."""
    pass


class VocabularyManager:
    """
    Manage custom vocabulary for improved transcription accuracy.

    This class maintains a vocabulary of domain-specific terms organized by
    category. The vocabulary is used to build an initial_prompt for Whisper,
    which improves recognition of specialized terminology.

    Attributes:
        vocab_path: Path to YAML vocabulary file
        vocabulary: Dictionary mapping categories to term lists

    Categories:
        - companies: Company and brand names (AdsWizz, Pandora, etc.)
        - technical_terms: Technical jargon (SFTP, TLS, API, etc.)
        - speaker_names: Known speaker names (synced from speaker DB)
        - custom: User-defined custom terms

    Design decisions:
        - YAML storage for easy manual editing
        - Category-based organization for maintainability
        - Deduplication to avoid redundant hints
        - Token-aware prompt building (respects Whisper's 224 token limit)
    """

    DEFAULT_CATEGORIES = ["companies", "technical_terms", "speaker_names", "custom"]
    MAX_PROMPT_LENGTH = 200  # Conservative limit to avoid token overflow

    def __init__(self, vocab_path: Path):
        """
        Initialize vocabulary manager.

        Args:
            vocab_path: Path to YAML vocabulary file

        Raises:
            VocabularyError: If vocabulary file is corrupted
        """
        self.vocab_path = vocab_path
        self.vocabulary: Dict[str, List[str]] = {
            cat: [] for cat in self.DEFAULT_CATEGORIES
        }

        # Load existing vocabulary
        self._load_vocabulary()

        logger.info(
            f"Initialized VocabularyManager: {self.total_terms()} terms across "
            f"{len(self.vocabulary)} categories"
        )

    def add_term(
        self,
        term: str,
        category: str = "custom",
        save: bool = True
    ) -> None:
        """
        Add a term to vocabulary.

        Args:
            term: Term to add (case-sensitive)
            category: Category for the term
            save: If True, save vocabulary to file immediately

        Raises:
            ValueError: If term is empty or category is invalid
        """
        if not term or not term.strip():
            raise ValueError("Term cannot be empty")

        term = term.strip()

        # Create category if it doesn't exist
        if category not in self.vocabulary:
            self.vocabulary[category] = []

        # Add if not already present (case-sensitive)
        if term not in self.vocabulary[category]:
            self.vocabulary[category].append(term)
            logger.debug(f"Added term '{term}' to category '{category}'")

            if save:
                self._save_vocabulary()
        else:
            logger.debug(f"Term '{term}' already exists in category '{category}'")

    def remove_term(
        self,
        term: str,
        category: Optional[str] = None,
        save: bool = True
    ) -> bool:
        """
        Remove a term from vocabulary.

        Args:
            term: Term to remove
            category: Category to remove from (if None, removes from all)
            save: If True, save vocabulary to file immediately

        Returns:
            True if term was removed, False if not found
        """
        removed = False

        if category:
            # Remove from specific category
            if category in self.vocabulary and term in self.vocabulary[category]:
                self.vocabulary[category].remove(term)
                logger.debug(f"Removed term '{term}' from category '{category}'")
                removed = True
        else:
            # Remove from all categories
            for cat in self.vocabulary:
                if term in self.vocabulary[cat]:
                    self.vocabulary[cat].remove(term)
                    logger.debug(f"Removed term '{term}' from category '{cat}'")
                    removed = True

        if removed and save:
            self._save_vocabulary()

        return removed

    def sync_speaker_names(self, speaker_names: List[str]) -> None:
        """
        Sync speaker names from speaker database.

        This ensures the vocabulary always reflects enrolled speakers,
        helping Whisper recognize their names correctly.

        Args:
            speaker_names: List of speaker names from database
        """
        # Replace speaker_names category entirely
        self.vocabulary["speaker_names"] = list(set(speaker_names))
        logger.info(f"Synced {len(speaker_names)} speaker names to vocabulary")
        self._save_vocabulary()

    def build_initial_prompt(self, max_length: int = MAX_PROMPT_LENGTH) -> str:
        """
        Build optimized initial_prompt for Whisper.

        The initial_prompt provides hints to Whisper about expected vocabulary,
        significantly improving accuracy for specialized terms. We build a
        natural-sounding sentence to stay within token limits.

        Args:
            max_length: Maximum prompt length in characters

        Returns:
            Optimized prompt string for Whisper

        Note:
            Whisper has a 224 token limit for initial_prompt. We use character
            count as a proxy (roughly 4 chars per token).
        """
        # Collect all terms, deduplicate
        all_terms: Set[str] = set()

        # Prioritize: companies > technical_terms > speaker_names > custom
        priority_order = ["companies", "technical_terms", "speaker_names", "custom"]

        for category in priority_order:
            if category in self.vocabulary:
                all_terms.update(self.vocabulary[category])

        if not all_terms:
            return ""

        # Build natural prompt sentence
        terms_list = sorted(all_terms)  # Sort for consistency

        # Start with a natural prefix
        prompt = "This transcript discusses "

        # Add terms until we hit the length limit
        terms_added = []
        for term in terms_list:
            test_prompt = prompt + ", ".join(terms_added + [term]) + "."

            if len(test_prompt) > max_length:
                break

            terms_added.append(term)

        if terms_added:
            prompt += ", ".join(terms_added) + "."
        else:
            prompt = ""

        logger.debug(f"Built initial_prompt: {len(prompt)} chars, {len(terms_added)} terms")
        return prompt

    def get_all_terms(self) -> Dict[str, List[str]]:
        """Get all terms organized by category."""
        return dict(self.vocabulary)

    def get_category_terms(self, category: str) -> List[str]:
        """
        Get terms for a specific category.

        Args:
            category: Category name

        Returns:
            List of terms in category
        """
        return self.vocabulary.get(category, [])

    def total_terms(self) -> int:
        """Get total number of terms across all categories."""
        return sum(len(terms) for terms in self.vocabulary.values())

    def clear_category(self, category: str, save: bool = True) -> None:
        """
        Clear all terms in a category.

        Args:
            category: Category to clear
            save: If True, save vocabulary to file immediately
        """
        if category in self.vocabulary:
            self.vocabulary[category] = []
            logger.info(f"Cleared category '{category}'")

            if save:
                self._save_vocabulary()

    def _load_vocabulary(self) -> None:
        """Load vocabulary from YAML file."""
        if not self.vocab_path.exists():
            logger.info(f"No existing vocabulary found at {self.vocab_path}")
            return

        try:
            with open(self.vocab_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("Vocabulary file is empty")
                return

            # Validate and load
            if not isinstance(data, dict):
                raise ValueError("Vocabulary must be a dictionary")

            # Merge with existing categories, ensuring lists
            for category, terms in data.items():
                if not isinstance(terms, list):
                    raise ValueError(f"Category '{category}' must contain a list")

                self.vocabulary[category] = terms

            logger.info(f"Loaded {self.total_terms()} terms from {self.vocab_path}")

        except yaml.YAMLError as e:
            raise VocabularyError(
                f"Corrupted vocabulary file at {self.vocab_path}: {e}"
            ) from e
        except Exception as e:
            raise VocabularyError(
                f"Failed to load vocabulary from {self.vocab_path}: {e}"
            ) from e

    def _save_vocabulary(self) -> None:
        """Save vocabulary to YAML file."""
        try:
            # Ensure directory exists
            self.vocab_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data (sort categories and terms for consistency)
            sorted_vocab = {}
            for category in sorted(self.vocabulary.keys()):
                if self.vocabulary[category]:  # Only save non-empty categories
                    sorted_vocab[category] = sorted(self.vocabulary[category])

            # Write atomically (write to temp, then rename)
            temp_path = self.vocab_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                yaml.safe_dump(
                    sorted_vocab,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )

            temp_path.replace(self.vocab_path)
            logger.debug(f"Saved vocabulary to {self.vocab_path}")

        except Exception as e:
            raise VocabularyError(
                f"Failed to save vocabulary to {self.vocab_path}: {e}"
            ) from e
