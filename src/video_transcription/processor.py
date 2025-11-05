"""Combined video processing - transcription and analysis."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .analyzer import VideoAnalyzer, VideoAnalysisResult
from .transcriber import VideoTranscriber, TranscriptionResult


class VideoProcessingResult(BaseModel):
    """Combined result of transcription and analysis."""

    transcription: TranscriptionResult
    analysis: VideoAnalysisResult
    combined_summary: Optional[str] = None


class VideoProcessor:
    """Process videos with both transcription and visual analysis."""

    def __init__(
        self,
        whisper_model: str = "medium",
        enable_diarization: bool = True,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2-vision",
    ):
        """
        Initialize the processor with MLX-accelerated transcription.

        Args:
            whisper_model: Whisper model size (MLX auto-uses GPU on Apple Silicon)
            enable_diarization: Enable speaker diarization (default: True)
            ollama_url: Ollama API URL
            ollama_model: Ollama vision model
        """
        self.transcriber = VideoTranscriber(
            model_name=whisper_model,
            enable_diarization=enable_diarization
        )
        self.analyzer = VideoAnalyzer(ollama_url=ollama_url, model=ollama_model)

    def process_video(
        self,
        video_path: Path,
        analysis_fps: float = 1.0,
        max_frames: Optional[int] = None,
        frame_prompt: str = "Describe what you see in this image in detail.",
        generate_combined_summary: bool = True,
    ) -> VideoProcessingResult:
        """
        Process video with both transcription and analysis.

        Args:
            video_path: Path to video file
            analysis_fps: FPS for frame extraction
            max_frames: Max frames to analyze
            frame_prompt: Prompt for frame analysis
            generate_combined_summary: Generate summary combining both

        Returns:
            VideoProcessingResult with transcription and analysis
        """
        print(f"\n{'='*60}")
        print(f"Processing video: {video_path}")
        print(f"{'='*60}\n")

        # Transcribe audio
        print("STEP 1: Transcription")
        print("-" * 60)
        transcription = self.transcriber.transcribe(video_path)
        print(f"✓ Transcribed {len(transcription.segments)} segments")
        print(f"✓ Language: {transcription.language}\n")

        # Analyze video frames
        print("STEP 2: Visual Analysis")
        print("-" * 60)
        analysis = self.analyzer.analyze_video(
            video_path,
            fps=analysis_fps,
            max_frames=max_frames,
            frame_prompt=frame_prompt,
            generate_summary=True,
        )
        print(f"✓ Analyzed {len(analysis.frames)} frames\n")

        # Generate combined summary
        combined_summary = None
        if generate_combined_summary:
            print("STEP 3: Combined Summary")
            print("-" * 60)
            combined_summary = self._generate_combined_summary(transcription, analysis)
            print("✓ Generated combined summary\n")

        print(f"{'='*60}")
        print("Processing complete!")
        print(f"{'='*60}\n")

        return VideoProcessingResult(
            transcription=transcription,
            analysis=analysis,
            combined_summary=combined_summary,
        )

    def _generate_combined_summary(
        self, transcription: TranscriptionResult, analysis: VideoAnalysisResult
    ) -> str:
        """Generate summary combining transcription and visual analysis."""
        prompt = f"""Analyze this video by combining the audio transcription and visual descriptions:

AUDIO TRANSCRIPTION:
{transcription.full_text}

VISUAL ANALYSIS:
{analysis.summary or 'No visual summary available'}

Provide a comprehensive summary that integrates both the audio and visual elements of the video:"""

        import requests

        payload = {
            "model": self.analyzer.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.analyzer.ollama_url}/api/generate", json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["response"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to generate combined summary: {e}")
            return "Combined summary generation failed"

    def save_results(
        self, result: VideoProcessingResult, output_dir: Path, base_name: str
    ) -> None:
        """
        Save all results to files.

        Args:
            result: Processing result
            output_dir: Output directory
            base_name: Base name for output files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save transcription
        self.transcriber.save_transcription(
            result.transcription, output_dir / f"{base_name}_transcript.txt", format="txt"
        )
        self.transcriber.save_transcription(
            result.transcription, output_dir / f"{base_name}_transcript.srt", format="srt"
        )
        self.transcriber.save_transcription(
            result.transcription,
            output_dir / f"{base_name}_transcript.json",
            format="json",
        )

        # Save analysis
        self.analyzer.save_analysis(
            result.analysis, output_dir / f"{base_name}_analysis.json", format="json"
        )
        self.analyzer.save_analysis(
            result.analysis, output_dir / f"{base_name}_analysis.md", format="md"
        )

        # Save combined summary
        if result.combined_summary:
            (output_dir / f"{base_name}_combined_summary.txt").write_text(
                result.combined_summary
            )

        print(f"✓ Saved all results to {output_dir}")
