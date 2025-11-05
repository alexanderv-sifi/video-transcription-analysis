#!/usr/bin/env python3
"""
Custom vocabulary management CLI tool.

Usage:
    # Add terms to vocabulary
    python manage_vocabulary.py add "AdsWizz" --category companies
    python manage_vocabulary.py add "SFTP" --category technical_terms
    python manage_vocabulary.py add "TLS 1.2" --category technical_terms

    # Remove a term
    python manage_vocabulary.py remove "AdsWizz"

    # List all vocabulary
    python manage_vocabulary.py list

    # List terms in a specific category
    python manage_vocabulary.py list --category companies

    # Clear a category
    python manage_vocabulary.py clear --category custom

    # Show the initial_prompt that will be built
    python manage_vocabulary.py preview
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_transcription.vocabulary import VocabularyManager


def add_term(args):
    """Add a term to vocabulary."""
    vocab = VocabularyManager(args.vocab_path)

    try:
        vocab.add_term(args.term, category=args.category)
        print(f"✓ Added '{args.term}' to category '{args.category}'")
        print(f"Total terms: {vocab.total_terms()}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def remove_term(args):
    """Remove a term from vocabulary."""
    vocab = VocabularyManager(args.vocab_path)

    if vocab.remove_term(args.term, category=args.category):
        if args.category:
            print(f"✓ Removed '{args.term}' from category '{args.category}'")
        else:
            print(f"✓ Removed '{args.term}' from all categories")
        print(f"Total terms: {vocab.total_terms()}")
    else:
        print(f"Error: Term '{args.term}' not found")
        sys.exit(1)


def list_vocabulary(args):
    """List vocabulary terms."""
    vocab = VocabularyManager(args.vocab_path)

    if args.category:
        # List specific category
        terms = vocab.get_category_terms(args.category)
        if not terms:
            print(f"No terms in category '{args.category}'")
            return

        print(f"\n{'='*60}")
        print(f"Category: {args.category} ({len(terms)} terms)")
        print(f"{'='*60}")
        for term in sorted(terms):
            print(f"  • {term}")
        print(f"{'='*60}")

    else:
        # List all categories
        all_terms = vocab.get_all_terms()
        total = vocab.total_terms()

        if total == 0:
            print("No terms in vocabulary.")
            return

        print(f"\n{'='*60}")
        print(f"Custom Vocabulary ({total} terms)")
        print(f"{'='*60}")

        for category in sorted(all_terms.keys()):
            terms = all_terms[category]
            if terms:
                print(f"\n{category} ({len(terms)}):")
                for term in sorted(terms):
                    print(f"  • {term}")

        print(f"\n{'='*60}")


def clear_category(args):
    """Clear all terms in a category."""
    vocab = VocabularyManager(args.vocab_path)

    if not args.force:
        response = input(f"Are you sure you want to clear category '{args.category}'? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    vocab.clear_category(args.category)
    print(f"✓ Cleared category '{args.category}'")


def preview_prompt(args):
    """Preview the initial_prompt that will be built."""
    vocab = VocabularyManager(args.vocab_path)

    prompt = vocab.build_initial_prompt()

    if not prompt:
        print("No terms in vocabulary. Initial prompt will be empty.")
        return

    print(f"\n{'='*60}")
    print(f"Initial Prompt Preview")
    print(f"{'='*60}")
    print(f"Length: {len(prompt)} characters")
    print(f"Terms included: {vocab.total_terms()}")
    print(f"\nPrompt:")
    print(f"{'-'*60}")
    print(prompt)
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage custom vocabulary for improved transcription"
    )

    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("vocabulary.yaml"),
        help="Path to vocabulary YAML (default: vocabulary.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a term to vocabulary")
    add_parser.add_argument("term", help="Term to add")
    add_parser.add_argument(
        "--category",
        type=str,
        default="custom",
        help="Category for the term (default: custom)",
    )

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a term from vocabulary")
    remove_parser.add_argument("term", help="Term to remove")
    remove_parser.add_argument(
        "--category",
        type=str,
        help="Remove only from this category (default: all categories)",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List vocabulary terms")
    list_parser.add_argument(
        "--category",
        type=str,
        help="List only this category (default: all categories)",
    )

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all terms in a category")
    clear_parser.add_argument("--category", type=str, required=True, help="Category to clear")
    clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # Preview command
    subparsers.add_parser("preview", help="Preview the initial_prompt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "add":
        add_term(args)
    elif args.command == "remove":
        remove_term(args)
    elif args.command == "list":
        list_vocabulary(args)
    elif args.command == "clear":
        clear_category(args)
    elif args.command == "preview":
        preview_prompt(args)


if __name__ == "__main__":
    main()
