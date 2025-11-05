#!/usr/bin/env python3
"""
Learn correction patterns from manually edited transcripts.

Usage:
    # Learn from transcript edits
    python learn_corrections.py original.txt edited.txt

    # Auto-approve all suggestions
    python learn_corrections.py original.txt edited.txt --auto-approve

    # Specify output files
    python learn_corrections.py original.txt edited.txt \
        --vocabulary vocab.yaml \
        --corrections corrections.yaml

    # Just analyze without updating files
    python learn_corrections.py original.txt edited.txt --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_transcription.corrector import DictionaryCorrector
from video_transcription.learning import CorrectionLearner
from video_transcription.vocabulary import VocabularyManager


def main():
    parser = argparse.ArgumentParser(
        description="Learn correction patterns from manually edited transcripts"
    )

    parser.add_argument("original", type=Path, help="Path to original transcript")
    parser.add_argument("edited", type=Path, help="Path to edited transcript")

    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=Path("vocabulary.yaml"),
        help="Path to vocabulary YAML (default: vocabulary.yaml)",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=Path("corrections.yaml"),
        help="Path to corrections YAML (default: corrections.yaml)",
    )

    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Minimum pattern occurrences to suggest (default: 2)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)",
    )

    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve all suggestions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show suggestions without updating files",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.original.exists():
        print(f"Error: Original transcript not found: {args.original}")
        sys.exit(1)

    if not args.edited.exists():
        print(f"Error: Edited transcript not found: {args.edited}")
        sys.exit(1)

    # Load transcripts
    original_text = args.original.read_text()
    edited_text = args.edited.read_text()

    print(f"\n{'='*60}")
    print(f"Learning Corrections from Manual Edits")
    print(f"{'='*60}")
    print(f"Original: {args.original}")
    print(f"Edited:   {args.edited}")
    print(f"{'='*60}\n")

    # Initialize learner
    learner = CorrectionLearner(
        min_occurrences=args.min_occurrences,
        min_confidence=args.min_confidence,
    )

    # Learn from diff
    try:
        result = learner.learn_from_diff(original_text, edited_text)
    except Exception as e:
        print(f"Error: Failed to analyze diff: {e}")
        sys.exit(1)

    # Display statistics
    print(f"Analysis Results:")
    print(f"  Total words: {result.total_words}")
    print(f"  Words changed: {result.words_changed} ({result.change_rate:.1%})")
    print(f"  Unique patterns: {len(result.pattern_frequency)}")
    print()

    # Generate suggestions
    suggestions = learner.suggest_rules(result)

    if not suggestions:
        print("No high-confidence suggestions found.")
        print("Try adjusting --min-occurrences or --min-confidence parameters.")
        return

    print(f"Found {len(suggestions)} suggestions:\n")
    print(f"{'-'*60}")

    # Group suggestions by type
    vocab_suggestions = [s for s in suggestions if s.type == "vocabulary"]
    dict_suggestions = [s for s in suggestions if s.type == "dictionary"]

    # Display and approve suggestions
    approved_vocab = []
    approved_dict = []

    if vocab_suggestions:
        print(f"\n📚 VOCABULARY SUGGESTIONS ({len(vocab_suggestions)}):")
        print("These terms will be added to initial_prompt for better recognition.\n")

        for i, suggestion in enumerate(vocab_suggestions, 1):
            print(f"{i}. Add term: '{suggestion.pattern}'")
            print(f"   Occurrences: {suggestion.occurrences}")
            print(f"   Confidence: {suggestion.confidence:.2f}")
            print(f"   Example: {suggestion.contexts[0]}")

            if args.auto_approve or args.dry_run:
                approved = args.auto_approve
            else:
                response = input(f"   Approve? (y/N): ")
                approved = response.lower() == 'y'

            if approved:
                approved_vocab.append(suggestion)
                print(f"   ✓ Approved")
            else:
                print(f"   ✗ Skipped")

            print()

    if dict_suggestions:
        print(f"\n📝 DICTIONARY SUGGESTIONS ({len(dict_suggestions)}):")
        print("These rules will automatically correct transcription errors.\n")

        for i, suggestion in enumerate(dict_suggestions, 1):
            print(f"{i}. '{suggestion.pattern}' → '{suggestion.replacement}'")
            print(f"   Occurrences: {suggestion.occurrences}")
            print(f"   Confidence: {suggestion.confidence:.2f}")
            print(f"   Example: {suggestion.contexts[0]}")

            if args.auto_approve or args.dry_run:
                approved = args.auto_approve
            else:
                response = input(f"   Approve? (y/N): ")
                approved = response.lower() == 'y'

            if approved:
                approved_dict.append(suggestion)
                print(f"   ✓ Approved")
            else:
                print(f"   ✗ Skipped")

            print()

    # Apply approved suggestions
    if not args.dry_run and (approved_vocab or approved_dict):
        print(f"{'-'*60}")
        print(f"Applying approved suggestions...\n")

        # Add to vocabulary
        if approved_vocab:
            vocab = VocabularyManager(args.vocabulary)
            for suggestion in approved_vocab:
                vocab.add_term(suggestion.pattern, category="custom")
            print(f"✓ Added {len(approved_vocab)} terms to vocabulary")

        # Add to corrections
        if approved_dict:
            corrector = DictionaryCorrector(args.corrections)
            for suggestion in approved_dict:
                corrector.add_rule(suggestion.pattern, suggestion.replacement)
            print(f"✓ Added {len(approved_dict)} rules to corrections")

        print(f"\n{'='*60}")
        print(f"Learning Complete!")
        print(f"{'='*60}")
        print(f"Vocabulary: {args.vocabulary}")
        print(f"Corrections: {args.corrections}")
        print(f"{'='*60}")

    elif args.dry_run:
        print(f"{'-'*60}")
        print(f"Dry run complete. No files were modified.")
        print(f"{'-'*60}")


if __name__ == "__main__":
    main()
