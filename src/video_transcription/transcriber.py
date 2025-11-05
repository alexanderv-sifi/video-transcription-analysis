"""Audio transcription using Whisper with speaker diarization."""

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ffmpeg
import mlx_whisper
import torch
from pydantic import BaseModel
from pyannote.audio import Pipeline

from .corrector import DictionaryCorrector
from .llm_corrector import LLMCorrector
from .speaker_db import SpeakerDatabase
from .vocabulary import VocabularyManager


class TranscriptionSegment(BaseModel):
    """A segment of transcribed text."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscriptionResult(BaseModel):
    """Complete transcription result."""

    segments: List[TranscriptionSegment]
    language: str
    full_text: str


class VideoTranscriber:
    """Transcribe audio from video files using Whisper with speaker diarization."""

    def __init__(
        self,
        model_name: str = "medium",
        enable_diarization: bool = True,
        hf_token: Optional[str] = None,
        speaker_db_path: Optional[Path] = None,
        vocabulary_path: Optional[Path] = None,
        corrections_path: Optional[Path] = None,
        enable_llm_corrections: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        domain_context: Optional[str] = None,
    ):
        """
        Initialize the transcriber with MLX-accelerated Whisper for Apple Silicon.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, turbo)
            enable_diarization: Enable speaker diarization with MPS acceleration
            hf_token: HuggingFace token for pyannote models (optional, will use HF_TOKEN env var)
            speaker_db_path: Path to speaker database JSON (enables speaker identification)
            vocabulary_path: Path to vocabulary YAML (improves recognition of custom terms)
            corrections_path: Path to corrections YAML (applies dictionary-based fixes)
            enable_llm_corrections: Enable LLM-based corrections via Ollama
            ollama_url: URL for Ollama API
            ollama_model: Ollama model for corrections
            domain_context: Optional domain context for LLM corrections

        Note:
            MLX automatically uses GPU acceleration on Apple Silicon (M1/M2/M3).
            Diarization pipeline will use MPS (Metal Performance Shaders) if available.
            Speaker identification, vocabulary, and corrections are optional enhancements.
            LLM corrections are the slowest tier and optional (requires Ollama running).
        """
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.diarization_pipeline: Optional[Pipeline] = None

        # Warn about tiny model quality issues
        if model_name == "tiny":
            print("\n" + "=" * 60)
            print("⚠ WARNING: Using 'tiny' model")
            print("=" * 60)
            print("The tiny model is ONLY for testing and has serious limitations:")
            print("  - High word error rate (7.5%)")
            print("  - Prone to hallucination (repetitive output)")
            print("  - Poor handling of technical terms")
            print("  - Unreliable for production use")
            print("\nRecommended models:")
            print("  - small:  Better quality, ~15 min for 1h video")
            print("  - medium: Best balance, ~25 min for 1h video (DEFAULT)")
            print("  - turbo:  Fast and good, ~20 min for 1h video")
            print("=" * 60 + "\n")

        # Optional enhancement modules (lazy-loaded)
        self.speaker_db_path = speaker_db_path
        self.vocabulary_path = vocabulary_path
        self.corrections_path = corrections_path
        self.enable_llm_corrections = enable_llm_corrections
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.domain_context = domain_context
        self._speaker_db: Optional[SpeakerDatabase] = None
        self._vocabulary: Optional[VocabularyManager] = None
        self._corrector: Optional[DictionaryCorrector] = None
        self._llm_corrector: Optional[LLMCorrector] = None

    @property
    def speaker_db(self) -> Optional[SpeakerDatabase]:
        """Lazy-load speaker database."""
        if self._speaker_db is None and self.speaker_db_path:
            print(f"Loading speaker database from {self.speaker_db_path}...")
            self._speaker_db = SpeakerDatabase(self.speaker_db_path)
        return self._speaker_db

    @property
    def vocabulary(self) -> Optional[VocabularyManager]:
        """Lazy-load vocabulary manager."""
        if self._vocabulary is None and self.vocabulary_path:
            print(f"Loading vocabulary from {self.vocabulary_path}...")
            self._vocabulary = VocabularyManager(self.vocabulary_path)
            # Sync speaker names if we have a speaker database
            if self.speaker_db:
                speaker_names = [p.name for p in self.speaker_db.list_speakers()]
                self._vocabulary.sync_speaker_names(speaker_names)
        return self._vocabulary

    @property
    def corrector(self) -> Optional[DictionaryCorrector]:
        """Lazy-load dictionary corrector."""
        if self._corrector is None and self.corrections_path:
            print(f"Loading corrections from {self.corrections_path}...")
            self._corrector = DictionaryCorrector(self.corrections_path)
        return self._corrector

    @property
    def llm_corrector(self) -> Optional[LLMCorrector]:
        """Lazy-load LLM corrector."""
        if self._llm_corrector is None and self.enable_llm_corrections:
            print(f"Initializing LLM corrector (Ollama: {self.ollama_model})...")
            self._llm_corrector = LLMCorrector(
                ollama_url=self.ollama_url,
                model=self.ollama_model
            )
        return self._llm_corrector

    def _load_diarization_pipeline(self) -> None:
        """Load the speaker diarization pipeline."""
        if self.enable_diarization and self.diarization_pipeline is None:
            print("Loading speaker diarization pipeline...")
            if not self.hf_token:
                print("WARNING: No HuggingFace token provided. Diarization may not work.")
                print("Set HF_TOKEN environment variable or pass hf_token parameter.")
                print("Get token from: https://huggingface.co/settings/tokens")
                self.enable_diarization = False
                return

            try:
                # pyannote.audio 3.x uses 'use_auth_token', 4.x uses 'token'
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )

                # Optimize for Apple Silicon MPS if available
                if torch.backends.mps.is_available():
                    print("✓ Moving diarization pipeline to MPS (GPU acceleration)...")
                    self.diarization_pipeline.to(torch.device("mps"))

                print("✓ Diarization pipeline loaded")
            except Exception as e:
                print(f"WARNING: Failed to load diarization pipeline: {e}")
                print("Continuing without speaker diarization...")
                self.enable_diarization = False

    def _assign_speakers(
        self, segments: List[TranscriptionSegment], audio_path: Path
    ) -> List[TranscriptionSegment]:
        """
        Assign speaker labels to transcription segments.

        Args:
            segments: List of transcription segments
            audio_path: Path to audio file

        Returns:
            Segments with speaker labels assigned
        """
        if not self.enable_diarization or not self.diarization_pipeline:
            return segments

        print("Performing speaker diarization...")
        try:
            # Run diarization
            output = self.diarization_pipeline(str(audio_path))

            # pyannote.audio 4.0+ uses .speaker_diarization attribute
            # older versions return an Annotation object directly with itertracks()
            if hasattr(output, 'speaker_diarization'):
                # pyannote.audio 4.0+ API
                diarization_iter = output.speaker_diarization
                print("Using pyannote.audio 4.0+ API (.speaker_diarization)")
            elif hasattr(output, 'itertracks'):
                # pyannote.audio 3.x API
                diarization_iter = output
                print("Using pyannote.audio 3.x API (.itertracks)")
            else:
                print(f"ERROR: Unexpected diarization output type: {type(output)}")
                print(f"Available attributes: {[m for m in dir(output) if not m.startswith('_')]}")
                return segments

            # Assign speakers to segments based on overlap
            for segment in segments:
                # Find speaker with most overlap in this segment
                segment_start = segment.start
                segment_end = segment.end
                segment_duration = segment_end - segment_start

                speaker_durations = {}

                # Iterate based on API version
                if hasattr(diarization_iter, 'itertracks'):
                    # 3.x API: itertracks yields (segment, track, speaker)
                    iterator = diarization_iter.itertracks(yield_label=True)
                    for diar_segment, _, speaker in iterator:
                        # Calculate overlap between diarization segment and transcription segment
                        overlap_start = max(diar_segment.start, segment_start)
                        overlap_end = min(diar_segment.end, segment_end)
                        overlap = max(0, overlap_end - overlap_start)

                        if overlap > 0:
                            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap
                else:
                    # 4.0+ API: speaker_diarization yields (turn, speaker)
                    for turn, speaker in diarization_iter:
                        # Calculate overlap between diarization turn and transcription segment
                        overlap_start = max(turn.start, segment_start)
                        overlap_end = min(turn.end, segment_end)
                        overlap = max(0, overlap_end - overlap_start)

                        if overlap > 0:
                            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap

                # Assign speaker with most overlap
                if speaker_durations:
                    segment.speaker = max(speaker_durations, key=speaker_durations.get)

            # Count speakers
            speakers = set(seg.speaker for seg in segments if seg.speaker)
            print(f"✓ Identified {len(speakers)} speakers: {', '.join(sorted(speakers))}")

        except Exception as e:
            print(f"WARNING: Diarization failed: {e}")
            print("Continuing without speaker labels...")

        return segments

    def _identify_speakers(
        self, segments: List[TranscriptionSegment], audio_path: Path
    ) -> List[TranscriptionSegment]:
        """
        Identify speakers using speaker database.

        This replaces generic SPEAKER_XX labels with actual names by matching
        voice embeddings against the speaker database.

        Args:
            segments: List of transcription segments with speaker labels
            audio_path: Path to audio file

        Returns:
            Segments with identified speaker names
        """
        if not self.speaker_db:
            return segments

        print("Identifying speakers from database...")

        # Group segments by speaker
        speaker_segments: Dict[str, List[TranscriptionSegment]] = {}
        for seg in segments:
            if seg.speaker:
                if seg.speaker not in speaker_segments:
                    speaker_segments[seg.speaker] = []
                speaker_segments[seg.speaker].append(seg)

        # Try to identify each speaker
        speaker_mapping: Dict[str, str] = {}  # SPEAKER_XX -> actual name

        for generic_speaker, segs in speaker_segments.items():
            # Find longest segment for this speaker (better for identification)
            longest_seg = max(segs, key=lambda s: s.end - s.start)

            # Extract audio segment
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_seg:
                    segment_path = Path(temp_seg.name)

                # Extract audio segment using ffmpeg
                (
                    ffmpeg.input(str(audio_path), ss=longest_seg.start, t=longest_seg.end - longest_seg.start)
                    .output(str(segment_path), acodec="pcm_s16le", ac=1, ar="16k")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )

                # Identify speaker
                name, confidence = self.speaker_db.identify_speaker(segment_path)

                if name:
                    speaker_mapping[generic_speaker] = name
                    print(f"  ✓ {generic_speaker} → {name} (confidence: {confidence:.2f})")
                else:
                    print(f"  ✗ {generic_speaker} → Unknown (no match)")

                # Cleanup
                if segment_path.exists():
                    segment_path.unlink()

            except Exception as e:
                print(f"  ✗ Failed to identify {generic_speaker}: {e}")

        # Apply speaker mapping
        if speaker_mapping:
            for seg in segments:
                if seg.speaker in speaker_mapping:
                    seg.speaker = speaker_mapping[seg.speaker]

            print(f"✓ Identified {len(speaker_mapping)} speakers from database")

        return segments

    def extract_audio(self, video_path: Path, audio_path: Path) -> None:
        """
        Extract audio from video file.

        Args:
            video_path: Path to input video
            audio_path: Path to output audio file
        """
        print(f"Extracting audio from {video_path}...")
        try:
            (
                ffmpeg.input(str(video_path))
                .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16k")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}") from e

    def transcribe(self, video_path: Path) -> TranscriptionResult:
        """
        Transcribe audio from video file with speaker diarization.

        Args:
            video_path: Path to video file

        Returns:
            TranscriptionResult with segments, speaker labels, and full text
        """
        if self.enable_diarization:
            self._load_diarization_pipeline()

        # Extract audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = Path(temp_audio.name)

        try:
            self.extract_audio(video_path, audio_path)

            # Build initial_prompt from vocabulary if available
            initial_prompt = ""
            if self.vocabulary:
                initial_prompt = self.vocabulary.build_initial_prompt()
                if initial_prompt:
                    print(f"✓ Using vocabulary hints ({len(initial_prompt)} chars)")

            # Build correct model path (naming is inconsistent in mlx-community)
            # tiny and turbo: no -mlx suffix
            # base, small, medium, large: -mlx suffix
            if self.model_name in ["tiny", "turbo"]:
                model_path = f"mlx-community/whisper-{self.model_name}"
            else:
                model_path = f"mlx-community/whisper-{self.model_name}-mlx"

            print(f"Transcribing audio with MLX-accelerated Whisper ({self.model_name})...")
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=model_path,
                verbose=False,
                initial_prompt=initial_prompt if initial_prompt else None
            )

            segments = [
                TranscriptionSegment(
                    start=seg["start"], end=seg["end"], text=seg["text"].strip()
                )
                for seg in result["segments"]
            ]

            # Assign speakers if diarization is enabled
            if self.enable_diarization and self.diarization_pipeline:
                segments = self._assign_speakers(segments, audio_path)

                # Try to identify speakers using speaker database
                if self.speaker_db:
                    segments = self._identify_speakers(segments, audio_path)

            # Build full text with speaker labels
            full_text = self._build_full_text(segments)

            # Detect and clean repetition (hallucination)
            has_repetition, full_text = self._detect_repetition(full_text)
            if has_repetition:
                print("⚠ Excessive repetition detected and truncated")
                print("  Consider using a larger model (small/medium) for better quality")

            # Validate transcription quality
            if not self._validate_transcription_quality(full_text):
                print("⚠ WARNING: Transcription quality appears low")
                print("  This may indicate hallucination or model issues")
                print("  Consider using a larger model or different audio preprocessing")

            # Apply dictionary corrections if available
            if self.corrector:
                full_text, stats = self.corrector.correct(full_text)
                if stats.total_corrections > 0:
                    print(f"✓ Applied {stats.total_corrections} dictionary corrections")

            # Apply LLM corrections if enabled
            if self.llm_corrector:
                try:
                    print("Applying LLM-based corrections (this may take a while)...")
                    full_text, llm_stats = self.llm_corrector.correct(
                        full_text, domain_context=self.domain_context
                    )
                    print(
                        f"✓ LLM corrections complete: {llm_stats.chunks_processed} chunks processed"
                    )
                    if llm_stats.failed_chunks > 0:
                        print(f"  ⚠ {llm_stats.failed_chunks} chunks failed (using original)")
                except Exception as e:
                    print(f"⚠ LLM corrections failed: {e}")
                    print("  Continuing with dictionary-corrected text...")

            return TranscriptionResult(
                segments=segments, language=result["language"], full_text=full_text
            )

        finally:
            # Cleanup temporary audio file
            if audio_path.exists():
                audio_path.unlink()

    @staticmethod
    def _detect_repetition(text: str, max_repetitions: int = 5) -> Tuple[bool, str]:
        """
        Detect excessive word repetition (hallucination) in text.

        Whisper models, especially smaller ones, can hallucinate by repeating
        the same word or phrase many times (e.g., "then then then then...").

        Args:
            text: Text to check for repetition
            max_repetitions: Maximum allowed consecutive repetitions of the same word

        Returns:
            Tuple of (has_repetition, cleaned_text)
            - has_repetition: True if excessive repetition was detected
            - cleaned_text: Text with excessive repetitions truncated
        """
        if not text or not text.strip():
            return False, text

        # Split into words
        words = text.split()
        if len(words) < max_repetitions:
            return False, text

        # Track consecutive repetitions
        has_repetition = False
        cleaned_words = []
        i = 0

        while i < len(words):
            word = words[i]

            # Count how many times this word repeats consecutively
            repetition_count = 1
            while (i + repetition_count < len(words) and
                   words[i + repetition_count].lower() == word.lower()):
                repetition_count += 1

            if repetition_count > max_repetitions:
                # Excessive repetition detected
                has_repetition = True
                # Keep only max_repetitions occurrences
                cleaned_words.extend([word] * max_repetitions)
                # Log the truncation
                print(f"⚠ Detected excessive repetition: '{word}' repeated {repetition_count} times (truncated to {max_repetitions})")
                i += repetition_count
            else:
                # Normal case - keep all occurrences
                cleaned_words.extend([word] * repetition_count)
                i += repetition_count

        cleaned_text = " ".join(cleaned_words)
        return has_repetition, cleaned_text

    @staticmethod
    def _validate_transcription_quality(text: str, min_unique_words: int = 10) -> bool:
        """
        Validate that transcription has reasonable quality.

        Low-quality transcriptions (hallucinations) often have:
        - Very few unique words
        - Excessive repetition
        - Very short length despite long audio

        Args:
            text: Transcribed text to validate
            min_unique_words: Minimum number of unique words expected

        Returns:
            True if quality seems acceptable, False otherwise
        """
        if not text or not text.strip():
            return False

        # Strip speaker labels for analysis
        text_without_labels = re.sub(r'(?:^|\n)[A-Z_]+\d*:\s*', ' ', text)

        words = text_without_labels.lower().split()
        if not words:
            return False

        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)

        # Low unique ratio suggests hallucination
        if unique_ratio < 0.1 and len(unique_words) < min_unique_words:
            print(f"⚠ Low transcription quality detected:")
            print(f"  Total words: {len(words)}")
            print(f"  Unique words: {len(unique_words)}")
            print(f"  Unique ratio: {unique_ratio:.2%}")
            return False

        return True

    def _build_full_text(self, segments: List[TranscriptionSegment]) -> str:
        """Build full text from segments with speaker labels."""
        if not segments:
            return ""

        # Check if any segments have speaker labels
        has_speakers = any(seg.speaker for seg in segments)

        if not has_speakers:
            return " ".join(seg.text for seg in segments)

        # Build text with speaker labels
        lines = []
        current_speaker = None

        for seg in segments:
            if seg.speaker != current_speaker:
                current_speaker = seg.speaker
                speaker_label = seg.speaker or "Unknown"
                lines.append(f"\n{speaker_label}: {seg.text}")
            else:
                lines.append(seg.text)

        return " ".join(lines).strip()

    def save_transcription(
        self, result: TranscriptionResult, output_path: Path, format: str = "txt"
    ) -> None:
        """
        Save transcription to file.

        Args:
            result: Transcription result
            output_path: Output file path
            format: Output format (txt, srt, json)
        """
        if format == "txt":
            output_path.write_text(result.full_text)
        elif format == "srt":
            self._save_as_srt(result, output_path)
        elif format == "json":
            output_path.write_text(result.model_dump_json(indent=2))
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_as_srt(self, result: TranscriptionResult, output_path: Path) -> None:
        """Save transcription as SRT subtitle file with speaker labels."""
        lines = []
        for i, seg in enumerate(result.segments, 1):
            start_time = self._format_timestamp(seg.start)
            end_time = self._format_timestamp(seg.end)

            # Add speaker label if available
            text = seg.text
            if seg.speaker:
                text = f"[{seg.speaker}] {text}"

            lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

        output_path.write_text("\n".join(lines))

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
