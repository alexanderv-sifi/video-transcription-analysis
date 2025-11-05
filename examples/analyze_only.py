#!/usr/bin/env python3
"""
Visual analysis only example.

Usage:
    python analyze_only.py /path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_transcription import VideoAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze video frames visually")
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to analyze (default: no limit)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-vision",
        help="Ollama model (default: llama3.2-vision)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe what you see in this image in detail.",
        help="Analysis prompt",
    )

    args = parser.parse_args()

    if not args.video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Initialize analyzer
    try:
        analyzer = VideoAnalyzer(ollama_url=args.ollama_url, model=args.model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Analyze video
    print(f"Analyzing {args.video_path}...")
    result = analyzer.analyze_video(
        args.video_path,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_prompt=args.prompt,
        generate_summary=True,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.video_path.stem

    analyzer.save_analysis(
        result, args.output_dir / f"{base_name}_analysis.json", format="json"
    )
    analyzer.save_analysis(
        result, args.output_dir / f"{base_name}_analysis.txt", format="txt"
    )
    analyzer.save_analysis(
        result, args.output_dir / f"{base_name}_analysis.md", format="md"
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")
    print(f"Frames analyzed: {len(result.frames)}")

    if result.summary:
        print(f"\nSummary:")
        print("-" * 60)
        print(result.summary)

    print(f"\nFrame-by-frame descriptions:")
    print("-" * 60)
    for frame in result.frames:
        print(f"\nFrame {frame.frame_number} (t={frame.timestamp:.2f}s):")
        print(f"  {frame.description[:200]}...")

    print(f"\n{'='*60}")
    print(f"Saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
