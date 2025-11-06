#!/usr/bin/env python3
"""
Audio transcription only example.

Usage:
    # Basic transcription with diarization
    python transcribe_only.py /path/to/video.mp4

    # With speaker identification
    python transcribe_only.py /path/to/video.mp4 --speaker-db speakers.json

    # With custom vocabulary and corrections
    python transcribe_only.py /path/to/video.mp4 --vocabulary vocab.yaml --corrections corrections.yaml

    # With LLM corrections (slow but context-aware)
    python transcribe_only.py /path/to/video.mp4 --corrections corrections.yaml --enable-llm-corrections --domain-context "Ad tech platform"

    # All features combined
    python transcribe_only.py /path/to/video.mp4 \
        --speaker-db speakers.json \
        --vocabulary vocab.yaml \
        --corrections corrections.yaml \
        --enable-llm-corrections \
        --domain-context "Ad tech platform"

    # Without diarization
    python transcribe_only.py /path/to/video.mp4 --no-diarization
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from video_transcription import VideoTranscriber

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio from video with MLX-accelerated Whisper (GPU on Apple Silicon)"
    )
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "turbo", "large-v3-turbo"],
        help="Whisper model size (default: medium, large-v3-turbo for 5x speed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization (enabled by default if HF_TOKEN is set)",
    )
    parser.add_argument(
        "--speaker-db",
        type=Path,
        help="Path to speaker database JSON (enables speaker identification)",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        help="Path to vocabulary YAML (improves recognition of custom terms)",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        help="Path to corrections YAML (applies dictionary-based fixes)",
    )
    parser.add_argument(
        "--enable-llm-corrections",
        action="store_true",
        help="Enable LLM-based corrections via Ollama (slow but context-aware)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.2",
        help="Ollama model for LLM corrections (default: llama3.2)",
    )
    parser.add_argument(
        "--domain-context",
        type=str,
        help="Domain context for LLM corrections (e.g., 'Ad tech platform')",
    )
    parser.add_argument(
        "--auto-detect-names",
        action="store_true",
        help="Auto-detect speaker names from video/transcript (Zoom/Teams labels, introductions)",
    )
    parser.add_argument(
        "--no-ocr-names",
        action="store_true",
        help="Disable OCR-based name detection from video labels",
    )
    parser.add_argument(
        "--no-ner-names",
        action="store_true",
        help="Disable NER-based name detection from transcript",
    )

    args = parser.parse_args()

    # Expand ~ in paths
    args.video_path = args.video_path.expanduser()
    if args.speaker_db:
        args.speaker_db = args.speaker_db.expanduser()
    if args.vocabulary:
        args.vocabulary = args.vocabulary.expanduser()
    if args.corrections:
        args.corrections = args.corrections.expanduser()
    args.output_dir = args.output_dir.expanduser()

    if not args.video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Initialize transcriber (MLX auto-uses GPU on Apple Silicon)
    # Speaker diarization is enabled by default
    transcriber = VideoTranscriber(
        model_name=args.model,
        enable_diarization=not args.no_diarization,
        speaker_db_path=args.speaker_db,
        vocabulary_path=args.vocabulary,
        corrections_path=args.corrections,
        enable_llm_corrections=args.enable_llm_corrections,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        domain_context=args.domain_context,
        auto_detect_names=args.auto_detect_names,
        enable_ocr_names=not args.no_ocr_names,
        enable_ner_names=not args.no_ner_names,
    )

    # Transcribe
    print(f"Transcribing {args.video_path}...")
    result = transcriber.transcribe(args.video_path)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.video_path.stem

    transcriber.save_transcription(
        result, args.output_dir / f"{base_name}_transcript.txt", format="txt"
    )
    transcriber.save_transcription(
        result, args.output_dir / f"{base_name}_transcript.srt", format="srt"
    )
    transcriber.save_transcription(
        result, args.output_dir / f"{base_name}_transcript.json", format="json"
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Transcription complete!")
    print(f"{'='*60}")
    print(f"Language: {result.language}")
    print(f"Segments: {len(result.segments)}")
    print(f"\nFull text:")
    print("-" * 60)
    print(result.full_text)
    print(f"\n{'='*60}")
    print(f"Saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
