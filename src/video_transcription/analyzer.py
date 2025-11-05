"""Video analysis using Ollama vision models."""

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests
from PIL import Image
from pydantic import BaseModel


class FrameAnalysis(BaseModel):
    """Analysis result for a single frame."""

    timestamp: float
    frame_number: int
    description: str
    metadata: Optional[Dict[str, Any]] = None


class VideoAnalysisResult(BaseModel):
    """Complete video analysis result."""

    frames: List[FrameAnalysis]
    summary: Optional[str] = None


class VideoAnalyzer:
    """Analyze video frames using Ollama vision models."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2-vision",
    ):
        """
        Initialize the analyzer.

        Args:
            ollama_url: Base URL for Ollama API
            model: Vision model to use
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify connection to Ollama."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Ollama at {self.ollama_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.ollama_url}. "
                f"Make sure Ollama is running. Error: {e}"
            ) from e

    def extract_frames(
        self, video_path: Path, fps: float = 1.0, max_frames: Optional[int] = None
    ) -> List[tuple[int, float, Image.Image]]:
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract

        Returns:
            List of (frame_number, timestamp, image) tuples
        """
        print(f"Extracting frames from {video_path} at {fps} fps...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)

        frames = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)

                    timestamp = frame_count / video_fps
                    frames.append((frame_count, timestamp, img))

                    if max_frames and len(frames) >= max_frames:
                        break

                frame_count += 1

        finally:
            cap.release()

        print(f"Extracted {len(frames)} frames")
        return frames

    def analyze_frame(
        self, image: Image.Image, prompt: str = "Describe what you see in this image in detail."
    ) -> str:
        """
        Analyze a single frame using Ollama vision model.

        Args:
            image: PIL Image to analyze
            prompt: Analysis prompt

        Returns:
            Analysis description
        """
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Call Ollama API
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["response"].strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to analyze frame: {e}") from e

    def analyze_video(
        self,
        video_path: Path,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
        frame_prompt: str = "Describe what you see in this image in detail.",
        generate_summary: bool = True,
    ) -> VideoAnalysisResult:
        """
        Analyze video by processing frames.

        Args:
            video_path: Path to video file
            fps: Frames per second to extract and analyze
            max_frames: Maximum number of frames to analyze
            frame_prompt: Prompt for analyzing each frame
            generate_summary: Whether to generate an overall summary

        Returns:
            VideoAnalysisResult with frame analyses and optional summary
        """
        frames = self.extract_frames(video_path, fps=fps, max_frames=max_frames)

        frame_analyses = []
        print(f"Analyzing {len(frames)} frames...")

        for i, (frame_num, timestamp, image) in enumerate(frames, 1):
            print(f"  Analyzing frame {i}/{len(frames)} (t={timestamp:.2f}s)...")
            description = self.analyze_frame(image, prompt=frame_prompt)

            frame_analyses.append(
                FrameAnalysis(
                    timestamp=timestamp,
                    frame_number=frame_num,
                    description=description,
                )
            )

        summary = None
        if generate_summary and frame_analyses:
            print("Generating overall summary...")
            summary = self._generate_summary(frame_analyses)

        return VideoAnalysisResult(frames=frame_analyses, summary=summary)

    def _generate_summary(self, frame_analyses: List[FrameAnalysis]) -> str:
        """Generate overall summary from frame analyses."""
        # Combine all frame descriptions
        descriptions = [
            f"At {fa.timestamp:.1f}s: {fa.description}" for fa in frame_analyses
        ]
        combined = "\n".join(descriptions)

        prompt = f"""Based on these frame-by-frame descriptions of a video, provide a concise summary of the overall video content, key events, and narrative:

{combined}

Summary:"""

        # Use text-only model for summary
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["response"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to generate summary: {e}")
            return "Summary generation failed"

    def save_analysis(
        self, result: VideoAnalysisResult, output_path: Path, format: str = "json"
    ) -> None:
        """
        Save analysis to file.

        Args:
            result: Analysis result
            output_path: Output file path
            format: Output format (json, txt, md)
        """
        if format == "json":
            output_path.write_text(result.model_dump_json(indent=2))
        elif format == "txt":
            self._save_as_txt(result, output_path)
        elif format == "md":
            self._save_as_markdown(result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_as_txt(self, result: VideoAnalysisResult, output_path: Path) -> None:
        """Save analysis as plain text."""
        lines = []

        if result.summary:
            lines.append("=== VIDEO SUMMARY ===\n")
            lines.append(f"{result.summary}\n\n")

        lines.append("=== FRAME-BY-FRAME ANALYSIS ===\n")
        for frame in result.frames:
            lines.append(f"\nFrame {frame.frame_number} (t={frame.timestamp:.2f}s):")
            lines.append(f"{frame.description}\n")

        output_path.write_text("\n".join(lines))

    def _save_as_markdown(self, result: VideoAnalysisResult, output_path: Path) -> None:
        """Save analysis as Markdown."""
        lines = ["# Video Analysis Report\n"]

        if result.summary:
            lines.append("## Summary\n")
            lines.append(f"{result.summary}\n")

        lines.append("## Frame-by-Frame Analysis\n")
        for frame in result.frames:
            lines.append(f"### Frame {frame.frame_number} (t={frame.timestamp:.2f}s)\n")
            lines.append(f"{frame.description}\n")

        output_path.write_text("\n".join(lines))
