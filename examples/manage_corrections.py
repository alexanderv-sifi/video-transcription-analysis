#!/usr/bin/env python3
"""
Dictionary corrections management CLI tool.

Usage:
    # Add a correction rule
    python manage_corrections.py add "as was" "AdsWizz"
    python manage_corrections.py add "Ed wiz" "AdsWizz"
    python manage_corrections.py add "TLS 1\.2" "TLS 1.2" --case-sensitive

    # Remove a correction rule
    python manage_corrections.py remove "as was"

    # List all correction rules
    python manage_corrections.py list

    # Clear all rules
    python manage_corrections.py clear

    # Test corrections on sample text
    python manage_corrections.py test "We use as was for the Ed wiz platform"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_transcription.corrector import DictionaryCorrector


def add_rule(args):
    """Add a correction rule."""
    corrector = DictionaryCorrector(args.corrections_path)

    try:
        corrector.add_rule(
            pattern=args.pattern,
            replacement=args.replacement,
            case_sensitive=args.case_sensitive
        )
        print(f"✓ Added correction: '{args.pattern}' → '{args.replacement}'")
        print(f"Total rules: {len(corrector.get_rules())}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def remove_rule(args):
    """Remove a correction rule."""
    corrector = DictionaryCorrector(args.corrections_path)

    if corrector.remove_rule(args.pattern):
        print(f"✓ Removed correction: '{args.pattern}'")
        print(f"Total rules: {len(corrector.get_rules())}")
    else:
        print(f"Error: Pattern '{args.pattern}' not found")
        sys.exit(1)


def list_rules(args):
    """List all correction rules."""
    corrector = DictionaryCorrector(args.corrections_path)
    rules = corrector.get_rules()

    if not rules:
        print("No correction rules defined.")
        return

    print(f"\n{'='*60}")
    print(f"Dictionary Corrections ({len(rules)} rules)")
    print(f"{'='*60}")

    for rule in rules:
        case_flag = " [case-sensitive]" if rule.case_sensitive else ""
        regex_flag = " [regex]" if rule.is_regex else ""
        print(f"\n  '{rule.pattern}' → '{rule.replacement}'{case_flag}{regex_flag}")

    print(f"\n{'='*60}")


def clear_rules(args):
    """Clear all correction rules."""
    corrector = DictionaryCorrector(args.corrections_path)

    if not args.force:
        response = input(f"Are you sure you want to clear all correction rules? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    corrector.clear_rules()
    print("✓ Cleared all correction rules")


def test_corrections(args):
    """Test corrections on sample text."""
    corrector = DictionaryCorrector(args.corrections_path)

    print(f"\n{'='*60}")
    print("Testing Corrections")
    print(f"{'='*60}")
    print(f"\nOriginal text:")
    print(f"{'-'*60}")
    print(args.text)

    corrected, stats = corrector.correct(args.text)

    print(f"\n{'-'*60}")
    print(f"Corrected text:")
    print(f"{'-'*60}")
    print(corrected)

    print(f"\n{'-'*60}")
    print(f"Statistics:")
    print(f"  Total corrections: {stats.total_corrections}")
    print(f"  Patterns matched: {len(stats.patterns_matched)}")
    if stats.patterns_matched:
        print(f"  Patterns: {', '.join(stats.patterns_matched)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage dictionary-based transcript corrections"
    )

    parser.add_argument(
        "--corrections-path",
        type=Path,
        default=Path("corrections.yaml"),
        help="Path to corrections YAML (default: corrections.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a correction rule")
    add_parser.add_argument("pattern", help="Pattern to match (supports regex)")
    add_parser.add_argument("replacement", help="Replacement text")
    add_parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Make matching case-sensitive",
    )

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a correction rule")
    remove_parser.add_argument("pattern", help="Pattern to remove")

    # List command
    subparsers.add_parser("list", help="List all correction rules")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all correction rules")
    clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test corrections on sample text")
    test_parser.add_argument("text", help="Text to test corrections on")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "add":
        add_rule(args)
    elif args.command == "remove":
        remove_rule(args)
    elif args.command == "list":
        list_rules(args)
    elif args.command == "clear":
        clear_rules(args)
    elif args.command == "test":
        test_corrections(args)


if __name__ == "__main__":
    main()
