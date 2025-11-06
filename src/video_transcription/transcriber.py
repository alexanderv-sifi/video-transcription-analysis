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
from .name_detector import SpeakerNameDetector
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
        auto_detect_names: bool = False,
        enable_ocr_names: bool = True,
        enable_ner_names: bool = True,
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
            auto_detect_names: Enable automatic speaker name detection from video/transcript
            enable_ocr_names: Use OCR to detect names from video labels (Zoom/Teams)
            enable_ner_names: Use NER to detect names from transcript

        Note:
            MLX automatically uses GPU acceleration on Apple Silicon (M1/M2/M3).
            Diarization pipeline will use MPS (Metal Performance Shaders) if available.
            Speaker identification, vocabulary, and corrections are optional enhancements.
            LLM corrections are the slowest tier and optional (requires Ollama running).
            Name detection finds speaker names automatically and can suggest enrollments.
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
        self.auto_detect_names = auto_detect_names
        self.enable_ocr_names = enable_ocr_names
        self.enable_ner_names = enable_ner_names
        self._speaker_db: Optional[SpeakerDatabase] = None
        self._vocabulary: Optional[VocabularyManager] = None
        self._corrector: Optional[DictionaryCorrector] = None
        self._llm_corrector: Optional[LLMCorrector] = None
        self._name_detector: Optional[SpeakerNameDetector] = None

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

    @property
    def name_detector(self) -> Optional[SpeakerNameDetector]:
        """Lazy-load name detector."""
        if self._name_detector is None and self.auto_detect_names:
            print("Initializing speaker name detector...")
            self._name_detector = SpeakerNameDetector(
                enable_ocr=self.enable_ocr_names,
                enable_ner=self.enable_ner_names
            )
        return self._name_detector

    def _load_diarization_pipeline(self) -> None:
        """Load the speaker diarization pipeline."""
        if self.enable_diarization and self.diarization_pipeline is None:
            print("Loading speaker diarization pipeline...")

            try:
                # Load Precision-2 pipeline (4.0) - 15% improvement over baseline
                # Uses cached models or HF_TOKEN from environment
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-4.0"
                )

                # Optimize for Apple Silicon MPS if available
                if torch.backends.mps.is_available():
                    print("✓ Moving diarization pipeline to MPS (GPU acceleration)...")
                    self.diarization_pipeline.to(torch.device("mps"))

                print("✓ Diarization pipeline loaded")
            except Exception as e:
                print(f"WARNING: Failed to load diarization pipeline: {e}")
                print("If models aren't cached, set HF_TOKEN in .env or run: huggingface-cli login")
                print("Get token from: https://huggingface.co/settings/tokens")
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
            # Load audio with torchaudio (avoids torchcodec FFmpeg dependency issue)
            import torchaudio

            waveform, sample_rate = torchaudio.load(str(audio_path))

            # Pass as dictionary (pyannote.audio 4.0 supports pre-loaded audio)
            audio_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            # Run diarization with pre-loaded audio
            output = self.diarization_pipeline(audio_dict)

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

    def _detect_and_map_names(
        self, segments: List[TranscriptionSegment], video_path: Path, audio_path: Path
    ) -> List[TranscriptionSegment]:
        """
        Detect speaker names and map them to speaker labels.

        Uses multi-source detection:
        1. OCR from video labels (Zoom/Teams names)
        2. NER from transcript (introductions, mentions)
        3. Pattern matching ("Hi, I'm Alex...")

        Then intelligently maps detected names to SPEAKER_XX labels
        and optionally offers to enroll new speakers.

        Args:
            segments: Transcription segments with speaker labels
            video_path: Path to video file
            audio_path: Path to extracted audio

        Returns:
            Segments with updated speaker names
        """
        if not self.name_detector:
            return segments

        print("\n" + "=" * 60)
        print("Auto-Detecting Speaker Names")
        print("=" * 60)

        all_detections = []

        # Detect from video labels (OCR)
        if self.enable_ocr_names:
            ocr_detections = self.name_detector.detect_from_video_labels(video_path)
            all_detections.extend(ocr_detections)

        # Detect from transcript (NER + patterns)
        if self.enable_ner_names:
            full_text = self._build_full_text(segments)
            ner_detections = self.name_detector.detect_from_transcript(full_text)
            all_detections.extend(ner_detections)

        if not all_detections:
            print("✗ No speaker names detected")
            print("=" * 60)
            return segments

        # Map names to speakers
        mappings = self.name_detector.map_names_to_speakers(segments, all_detections)

        # Print summary
        self.name_detector.print_detection_summary(all_detections, mappings)

        # Apply mappings to segments
        if mappings:
            for seg in segments:
                if seg.speaker in mappings:
                    mapping = mappings[seg.speaker]
                    seg.speaker = mapping.name

            # Check for unknown speakers and offer enrollment
            self._offer_speaker_enrollment(segments, mappings, audio_path)

        return segments

    def _offer_speaker_enrollment(
        self,
        segments: List[TranscriptionSegment],
        mappings: Dict[str, any],  # Dict[str, SpeakerNameMapping]
        audio_path: Path,
    ) -> None:
        """
        Offer to enroll newly detected speakers in the database.

        Args:
            segments: Transcription segments
            mappings: Speaker name mappings
            audio_path: Path to audio file
        """
        if not self.speaker_db:
            return

        print("\n" + "=" * 60)
        print("Speaker Enrollment")
        print("=" * 60)

        for speaker_label, mapping in mappings.items():
            # Check if this speaker is already in database
            # Extract a sample segment for this speaker
            speaker_segments = [s for s in segments if s.speaker == mapping.name]
            if not speaker_segments:
                continue

            # Use longest segment for identification
            longest_seg = max(speaker_segments, key=lambda s: s.end - s.start)

            # Extract audio segment
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_seg:
                    segment_path = Path(temp_seg.name)

                # Extract audio segment using ffmpeg
                (
                    ffmpeg.input(
                        str(audio_path), ss=longest_seg.start, t=longest_seg.end - longest_seg.start
                    )
                    .output(str(segment_path), acodec="pcm_s16le", ac=1, ar="16k")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )

                # Check for voice match conflict (same voice, different name)
                conflict = self.speaker_db.detect_voice_match_conflict(
                    mapping.name, segment_path
                )

                if conflict:
                    # Voice matches existing speaker but name differs
                    existing_uuid, existing_name, conf = conflict

                    # Try auto-resolution with alias
                    if self.speaker_db.auto_resolve_alias(mapping.name, existing_uuid, conf):
                        print(
                            f"\n✓ '{mapping.name}' recognized as '{existing_name}' "
                            f"(alias auto-created, {conf:.0%} confidence)"
                        )
                        # Update segments to use canonical name
                        for seg in speaker_segments:
                            seg.speaker = existing_name
                    else:
                        # Confidence too low for auto-resolution
                        print(
                            f"\n⚠ Voice similarity detected for '{mapping.name}':"
                        )
                        print(f"  Detected name: {mapping.name}")
                        print(f"  Similar to: {existing_name} ({conf:.0%} confidence)")
                        print(f"  Not auto-resolved (confidence < {self.speaker_db.conflict_config.alias_auto_threshold:.0%})")
                        print(f"  You may want to manually create an alias if these are the same person.")

                elif self.speaker_db.get_speaker_by_name(mapping.name):
                    # Speaker already exists with this exact name - skip enrollment
                    print(f"\n✓ '{mapping.name}' already enrolled")

                else:
                    # Check for name collision (different voices, same name)
                    collision = self.speaker_db.detect_name_collision(mapping.name, segment_path)

                    enrollment_name = mapping.name

                    if collision:
                        # Try auto-disambiguation
                        disambiguated = self.speaker_db.auto_resolve_disambiguation(
                            mapping.name, collision, segment_path
                        )

                        if disambiguated:
                            enrollment_name = disambiguated
                            print(
                                f"\n⚠ Another speaker named '{mapping.name}' exists with different voice"
                            )
                            print(f"  Auto-disambiguating as: '{enrollment_name}'")
                        else:
                            # Can't auto-resolve - warn user
                            print(
                                f"\n⚠ Name collision detected for '{mapping.name}':"
                            )
                            print(f"  Another speaker with this name already exists")
                            print(f"  Voice similarity too ambiguous for auto-disambiguation")
                            print(f"  Skipping enrollment - consider using a unique name")
                            continue

                    # New speaker - auto-enroll with best segments
                    print(f"\n🆕 New speaker detected: '{enrollment_name}'")
                    print(f"   Confidence: {mapping.confidence:.0%}")
                    print(f"   Auto-enrolling for future recognition...")

                    # Collect best segments for enrollment (3-5 diverse samples)
                    enrollment_segments = self._select_enrollment_segments(
                        speaker_segments, target_count=3
                    )

                    # Enroll speaker with selected segments
                    try:
                        enrollment_paths = []
                        for i, seg in enumerate(enrollment_segments):
                            with tempfile.NamedTemporaryFile(
                                suffix=f"_{i}.wav", delete=False
                            ) as temp_enroll:
                                enroll_path = Path(temp_enroll.name)

                            # Extract segment
                            (
                                ffmpeg.input(str(audio_path), ss=seg.start, t=seg.end - seg.start)
                                .output(str(enroll_path), acodec="pcm_s16le", ac=1, ar="16k")
                                .overwrite_output()
                                .run(capture_stdout=True, capture_stderr=True, quiet=True)
                            )
                            enrollment_paths.append(enroll_path)

                        # Enroll with all samples (use disambiguated name if needed)
                        self.speaker_db.enroll_speaker(enrollment_name, enrollment_paths)
                        print(f"   ✓ Enrolled '{enrollment_name}' with {len(enrollment_paths)} voice samples")
                        print(f"   (Will be recognized automatically in future calls)")

                        # Cleanup enrollment files
                        for path in enrollment_paths:
                            if path.exists():
                                path.unlink()

                    except Exception as e:
                        print(f"   ⚠ Auto-enrollment failed: {e}")
                        print(f"   To enroll manually later, use:")
                        print(f"   python examples/enroll_speaker.py enroll \"{mapping.name}\" <audio_samples>")

                # Cleanup
                if segment_path.exists():
                    segment_path.unlink()

            except Exception as e:
                print(f"  ✗ Failed to check enrollment for {mapping.name}: {e}")

        print("=" * 60)

    def _select_enrollment_segments(
        self, segments: List[TranscriptionSegment], target_count: int = 3
    ) -> List[TranscriptionSegment]:
        """
        Select best segments for speaker enrollment.

        Chooses diverse, high-quality segments spread throughout the recording.

        Criteria:
        - Segment length (prefer 10-30 seconds)
        - Temporal diversity (spread across recording)
        - Speech density (prefer segments with more speech)

        Args:
            segments: All segments for this speaker
            target_count: Number of segments to select

        Returns:
            List of selected segments for enrollment
        """
        if not segments:
            return []

        # Filter segments by length (10-30 seconds ideal)
        good_segments = []
        for seg in segments:
            duration = seg.end - seg.start
            if 10 <= duration <= 30:
                good_segments.append((seg, duration))
            elif 5 <= duration < 10:
                # Acceptable but not ideal
                good_segments.append((seg, duration * 0.8))
            elif 30 < duration <= 60:
                # Too long but usable
                good_segments.append((seg, duration * 0.6))

        if not good_segments:
            # Fall back to any segments
            good_segments = [(seg, seg.end - seg.start) for seg in segments]

        # Sort by quality score (duration-based)
        good_segments.sort(key=lambda x: x[1], reverse=True)

        # Select diverse segments (spread across time)
        if len(good_segments) <= target_count:
            return [seg for seg, _ in good_segments]

        # Divide recording into time buckets and pick best from each
        segments_by_time = sorted(good_segments, key=lambda x: x[0].start)
        selected = []

        bucket_size = len(segments_by_time) // target_count
        for i in range(target_count):
            bucket_start = i * bucket_size
            bucket_end = bucket_start + bucket_size if i < target_count - 1 else len(segments_by_time)
            bucket = segments_by_time[bucket_start:bucket_end]

            if bucket:
                # Pick best segment from this time bucket
                best = max(bucket, key=lambda x: x[1])
                selected.append(best[0])

        return selected

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
            # large-v3-turbo: special case with explicit version
            if self.model_name in ["tiny", "turbo"]:
                model_path = f"mlx-community/whisper-{self.model_name}"
            elif self.model_name == "large-v3-turbo":
                model_path = "mlx-community/whisper-large-v3-turbo"
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

                # Auto-detect speaker names from video/transcript
                if self.name_detector:
                    segments = self._detect_and_map_names(segments, video_path, audio_path)

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
