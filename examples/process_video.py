#!/usr/bin/env python3
"""
Complete video processing example with transcription and analysis.

Usage:
    python process_video.py /path/to/video.mp4
    python process_video.py /path/to/video.mp4 --fps 0.5 --max-frames 20
    python process_video.py /path/to/video.mp4 --no-diarization
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from video_transcription import VideoProcessor

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process video with MLX-accelerated transcription and visual analysis"
    )
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract for analysis (default: 1.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to analyze (default: no limit)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "turbo", "large-v3-turbo"],
        help="Whisper model size (default: medium, large-v3-turbo for 5x speed)",
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
        default="llama3.2-vision",
        help="Ollama vision model (default: llama3.2-vision)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--frame-prompt",
        type=str,
        default="Describe what you see in this image in detail.",
        help="Prompt for frame analysis",
    )
    parser.add_argument(
        "--no-combined-summary",
        action="store_true",
        help="Skip generating combined summary",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization (enabled by default if HF_TOKEN is set)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Initialize processor (MLX auto-uses GPU on Apple Silicon)
    # Speaker diarization is enabled by default
    try:
        processor = VideoProcessor(
            whisper_model=args.whisper_model,
            enable_diarization=not args.no_diarization,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
        )
    except Exception as e:
        print(f"Error initializing processor: {e}")
        sys.exit(1)

    # Process video
    try:
        result = processor.process_video(
            args.video_path,
            analysis_fps=args.fps,
            max_frames=args.max_frames,
            frame_prompt=args.frame_prompt,
            generate_combined_summary=not args.no_combined_summary,
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)

    # Save results
    base_name = args.video_path.stem
    try:
        processor.save_results(result, args.output_dir, base_name)
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTranscription ({len(result.transcription.segments)} segments):")
    print("-" * 60)
    print(result.transcription.full_text[:500] + "..." if len(result.transcription.full_text) > 500 else result.transcription.full_text)

    if result.analysis.summary:
        print(f"\nVisual Summary ({len(result.analysis.frames)} frames analyzed):")
        print("-" * 60)
        print(result.analysis.summary)

    if result.combined_summary:
        print("\nCombined Summary:")
        print("-" * 60)
        print(result.combined_summary)

    print("\n" + "=" * 60)
    print(f"All results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
