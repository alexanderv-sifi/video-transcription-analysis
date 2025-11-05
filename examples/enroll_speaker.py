#!/usr/bin/env python3
"""
Speaker enrollment and management CLI tool.

Usage:
    # Enroll a new speaker with audio samples
    python enroll_speaker.py enroll "Alexander" sample1.wav sample2.wav sample3.wav

    # List all enrolled speakers
    python enroll_speaker.py list

    # Remove a speaker
    python enroll_speaker.py remove "Alexander"

    # View speaker details
    python enroll_speaker.py info "Alexander"

    # View database statistics
    python enroll_speaker.py stats
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_transcription.speaker_db import SpeakerDatabase


def enroll_speaker(args):
    """Enroll a new speaker."""
    db = SpeakerDatabase(args.db_path)

    # Validate audio samples
    audio_samples = [Path(p) for p in args.audio_samples]
    for audio_path in audio_samples:
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

    try:
        profile = db.enroll_speaker(
            name=args.name,
            audio_samples=audio_samples,
            force=args.force
        )

        print(f"\n{'='*60}")
        print(f"✓ Speaker enrolled successfully!")
        print(f"{'='*60}")
        print(f"Name: {profile.name}")
        print(f"Samples: {profile.sample_count}")
        print(f"Enrolled: {profile.enrolled_date}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def list_speakers(args):
    """List all enrolled speakers."""
    db = SpeakerDatabase(args.db_path)
    speakers = db.list_speakers()

    if not speakers:
        print("No speakers enrolled in database.")
        return

    print(f"\n{'='*60}")
    print(f"Enrolled Speakers ({len(speakers)})")
    print(f"{'='*60}")

    for profile in sorted(speakers, key=lambda p: p.name):
        print(f"\n{profile.name}")
        print(f"  Enrolled: {profile.enrolled_date}")
        print(f"  Samples: {profile.sample_count}")
        if profile.last_seen:
            print(f"  Last seen: {profile.last_seen}")

    print(f"\n{'='*60}")


def remove_speaker(args):
    """Remove a speaker from database."""
    db = SpeakerDatabase(args.db_path)

    if not args.force:
        response = input(f"Are you sure you want to remove '{args.name}'? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    if db.remove_speaker(args.name):
        print(f"✓ Removed speaker: {args.name}")
    else:
        print(f"Error: Speaker not found: {args.name}")
        sys.exit(1)


def show_info(args):
    """Show detailed information about a speaker."""
    db = SpeakerDatabase(args.db_path)
    speakers = {s.name: s for s in db.list_speakers()}

    if args.name not in speakers:
        print(f"Error: Speaker not found: {args.name}")
        sys.exit(1)

    profile = speakers[args.name]

    print(f"\n{'='*60}")
    print(f"Speaker: {profile.name}")
    print(f"{'='*60}")
    print(f"Enrolled: {profile.enrolled_date}")
    print(f"Samples: {profile.sample_count}")
    print(f"Embedding dimension: {len(profile.embedding)}")
    if profile.last_seen:
        print(f"Last seen: {profile.last_seen}")
    print(f"{'='*60}")


def show_stats(args):
    """Show database statistics."""
    db = SpeakerDatabase(args.db_path)
    stats = db.get_stats()

    print(f"\n{'='*60}")
    print(f"Speaker Database Statistics")
    print(f"{'='*60}")
    print(f"Total speakers: {stats['total_speakers']}")
    print(f"Model: {stats['model']}")
    print(f"Threshold: {stats['threshold']}")
    print(f"Device: {stats['device']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Database path: {args.db_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage speaker database for voice identification"
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("speakers.json"),
        help="Path to speaker database JSON (default: speakers.json)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new speaker")
    enroll_parser.add_argument("name", help="Speaker name")
    enroll_parser.add_argument(
        "audio_samples",
        nargs="+",
        help="Audio file paths (3-5 samples recommended, 10-30 seconds each)",
    )
    enroll_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite if speaker already exists",
    )

    # List command
    subparsers.add_parser("list", help="List all enrolled speakers")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a speaker")
    remove_parser.add_argument("name", help="Speaker name to remove")
    remove_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show speaker details")
    info_parser.add_argument("name", help="Speaker name")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "enroll":
        enroll_speaker(args)
    elif args.command == "list":
        list_speakers(args)
    elif args.command == "remove":
        remove_speaker(args)
    elif args.command == "info":
        show_info(args)
    elif args.command == "stats":
        show_stats(args)


if __name__ == "__main__":
    main()
