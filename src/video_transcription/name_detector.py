"""Speaker name detection from multiple sources."""

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from pydantic import BaseModel


@dataclass
class NameDetection:
    """A detected name with metadata."""

    name: str
    source: str  # "ocr", "ner", "introduction"
    confidence: float  # 0.0 to 1.0
    timestamp: Optional[float] = None  # When detected (seconds)
    context: Optional[str] = None  # Surrounding text/info


@dataclass
class SpeakerNameMapping:
    """Mapping between speaker label and detected name."""

    speaker_label: str  # SPEAKER_00, SPEAKER_01, etc.
    name: str
    confidence: float
    detections: List[NameDetection]  # Supporting evidence


class SpeakerNameDetector:
    """
    Multi-source speaker name detection.

    Detects speaker names from:
    1. OCR of video labels (Zoom/Teams name overlays)
    2. NER from transcript (person names mentioned)
    3. Introduction patterns ("Hi, I'm Alex...")
    4. Common meeting patterns ("Thanks Alex for...")

    Then intelligently maps detected names to speaker labels
    based on temporal proximity and context.
    """

    # Common introduction patterns
    INTRO_PATTERNS = [
        r"(?:hi|hello|hey)[\s,]+(?:i'm|i am|this is|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:i'm|i am|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:^|\n)([A-Z][a-z]+):\s+(?:hi|hello|hey|good morning|good afternoon)",
    ]

    # Common guest/placeholder labels to ignore
    IGNORE_NAMES = {
        "guest",
        "unknown",
        "participant",
        "attendee",
        "user",
        "iphone",
        "ipad",
        "android",
        "mobile",
        "web",
        "phone",
        "tablet",
    }

    def __init__(self, enable_ocr: bool = True, enable_ner: bool = True):
        """
        Initialize name detector.

        Args:
            enable_ocr: Enable OCR-based name detection from video
            enable_ner: Enable NER-based name detection from transcript
        """
        self.enable_ocr = enable_ocr
        self.enable_ner = enable_ner
        self._nlp = None  # Lazy-loaded spaCy model

    @property
    def nlp(self):
        """Lazy-load spaCy NER model."""
        if self._nlp is None and self.enable_ner:
            try:
                import spacy

                print("Loading spaCy NER model...")
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("Downloading spaCy model en_core_web_sm...")
                    import subprocess

                    subprocess.run(
                        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                        check=True,
                    )
                    self._nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy model loaded")
            except Exception as e:
                print(f"⚠ Failed to load spaCy: {e}")
                self.enable_ner = False
        return self._nlp

    def detect_from_transcript(self, text: str) -> List[NameDetection]:
        """
        Extract person names from transcript using NER and patterns.

        Args:
            text: Full transcript text

        Returns:
            List of detected names with metadata
        """
        detections: List[NameDetection] = []

        # Method 1: Introduction patterns (high confidence)
        for pattern in self.INTRO_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                if self._is_valid_name(name):
                    detections.append(
                        NameDetection(
                            name=name,
                            source="introduction",
                            confidence=0.9,
                            context=text[max(0, match.start() - 50) : match.end() + 50],
                        )
                    )

        # Method 2: NER extraction (medium confidence)
        if self.enable_ner and self.nlp:
            try:
                doc = self.nlp(text[:10000])  # Limit for performance
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        name = ent.text.strip()
                        if self._is_valid_name(name):
                            detections.append(
                                NameDetection(
                                    name=name,
                                    source="ner",
                                    confidence=0.6,
                                    context=text[max(0, ent.start_char - 50) : ent.end_char + 50],
                                )
                            )
            except Exception as e:
                print(f"⚠ NER extraction failed: {e}")

        return detections

    def detect_from_video_labels(
        self, video_path: Path, sample_fps: float = 1.0
    ) -> List[NameDetection]:
        """
        Extract names from video using OCR on Zoom/Teams labels.

        Samples frames and runs OCR to find text in typical name label regions.

        Args:
            video_path: Path to video file
            sample_fps: Frames per second to sample (default: 0.5 = every 2 seconds)

        Returns:
            List of detected names with timestamps
        """
        if not self.enable_ocr:
            return []

        # Check if ocrmac is available (Mac only)
        try:
            import ocrmac
        except ImportError:
            print("⚠ ocrmac not available - skipping video OCR")
            print("  Install with: pip install ocrmac")
            return []

        detections: List[NameDetection] = []

        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(fps / sample_fps)

            print(f"Scanning video for name labels (sampling at {sample_fps} FPS)...")

            frame_num = 0
            all_text_found = []  # Debug: track all OCR text
            while frame_num < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_num / fps

                # Focus on top 20% of frame where Zoom/Teams labels typically appear
                height = frame.shape[0]
                top_region = frame[0:int(height * 0.2), :]

                # Extract text from frame using OCR
                text_items = self._ocr_frame(top_region)

                # Debug: collect all text for analysis
                if text_items:
                    all_text_found.extend(text_items)

                # Filter for potential names
                for text in text_items:
                    text = text.strip()
                    if self._is_valid_name(text):
                        detections.append(
                            NameDetection(
                                name=text,
                                source="ocr",
                                confidence=0.7,
                                timestamp=timestamp,
                            )
                        )

                frame_num += frame_interval

            # Debug: show what OCR found (even if not valid names)
            if all_text_found:
                unique_text = set(all_text_found)
                print(f"  OCR detected {len(unique_text)} unique text items:")
                for text in sorted(unique_text)[:20]:  # Show first 20
                    print(f"    - '{text}'")

            cap.release()

            # Deduplicate and count occurrences
            name_counts = Counter(d.name for d in detections)
            print(f"✓ Found {len(name_counts)} unique names in video labels")
            for name, count in name_counts.most_common(10):
                print(f"  - {name}: {count} occurrences")

        except Exception as e:
            print(f"⚠ Video OCR failed: {e}")

        return detections

    def _ocr_frame(self, frame: np.ndarray) -> List[str]:
        """
        Run OCR on video frame to extract text.

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            List of text strings found in frame
        """
        try:
            from ocrmac import ocrmac as ocr_module
            from PIL import Image, ImageEnhance

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Upscale for better OCR (2x)
            height, width = frame_rgb.shape[:2]
            frame_rgb = cv2.resize(frame_rgb, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Enhance contrast to make small text more readable
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)

            # Run OCR using Apple Vision framework
            # Returns: List[(text, confidence, (x, y, width, height))]
            results = ocr_module.text_from_image(
                pil_image,
                recognition_level="accurate",
                confidence_threshold=0.3  # Filter low-confidence results
            )

            # Extract text from results
            texts = []
            for text, confidence, bbox in results:
                text = text.strip()
                if text:
                    texts.append(text)

            return texts

        except Exception as e:
            # Silently fail for individual frames
            return []

    def _is_valid_name(self, name: str) -> bool:
        """
        Check if extracted text looks like a valid person name.

        Args:
            name: Candidate name string

        Returns:
            True if name appears valid
        """
        if not name or len(name) < 2:
            return False

        # Lowercase for checking
        name_lower = name.lower()

        # Ignore common non-names
        if name_lower in self.IGNORE_NAMES:
            return False

        # Must contain at least one letter
        if not any(c.isalpha() for c in name):
            return False

        # Must start with capital letter (proper name)
        if not name[0].isupper():
            return False

        # Ignore if too long (likely a sentence)
        if len(name) > 50:
            return False

        # Ignore if contains numbers (likely not a name)
        if any(c.isdigit() for c in name):
            return False

        # Split into words
        words = name.split()
        if len(words) > 4:  # Too many words
            return False

        # Check each word starts with capital
        for word in words:
            if not word[0].isupper():
                return False

        return True

    def map_names_to_speakers(
        self,
        segments: List,  # List[TranscriptionSegment]
        detections: List[NameDetection],
    ) -> Dict[str, SpeakerNameMapping]:
        """
        Map detected names to speaker labels based on temporal proximity.

        Args:
            segments: Transcription segments with speaker labels
            detections: Detected names from all sources

        Returns:
            Dict mapping speaker_label -> SpeakerNameMapping
        """
        if not detections:
            return {}

        # Build mapping: speaker_label -> candidate names with scores
        speaker_candidates: Dict[str, Dict[str, List[NameDetection]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for detection in detections:
            # Find segments near this detection
            nearby_segments = self._find_nearby_segments(segments, detection)

            for seg in nearby_segments:
                if seg.speaker:
                    speaker_candidates[seg.speaker][detection.name].append(detection)

        # For each speaker, pick the best name
        mappings: Dict[str, SpeakerNameMapping] = {}

        for speaker_label, candidates in speaker_candidates.items():
            if not candidates:
                continue

            # Score each candidate name
            name_scores: Dict[str, float] = {}
            for name, detections_list in candidates.items():
                # Score based on:
                # 1. Number of detections
                # 2. Average confidence
                # 3. Source diversity (more sources = better)

                count = len(detections_list)
                avg_confidence = sum(d.confidence for d in detections_list) / count
                unique_sources = len(set(d.source for d in detections_list))

                # Weighted score
                score = (count * 0.4) + (avg_confidence * 0.4) + (unique_sources * 0.2)
                name_scores[name] = score

            # Pick best name
            if name_scores:
                best_name = max(name_scores, key=name_scores.get)
                best_score = name_scores[best_name]

                # Normalize score to 0-1
                confidence = min(1.0, best_score / 3.0)

                mappings[speaker_label] = SpeakerNameMapping(
                    speaker_label=speaker_label,
                    name=best_name,
                    confidence=confidence,
                    detections=candidates[best_name],
                )

        return mappings

    def _find_nearby_segments(
        self, segments: List, detection: NameDetection, window_seconds: float = 30.0
    ) -> List:
        """
        Find segments near a name detection (temporal proximity).

        Args:
            segments: All transcription segments
            detection: Name detection with optional timestamp
            window_seconds: Time window to search (±seconds)

        Returns:
            List of segments near the detection
        """
        if detection.timestamp is None:
            # No timestamp - check if name appears in segment text
            nearby = []
            for seg in segments:
                if detection.name.lower() in seg.text.lower():
                    nearby.append(seg)
            return nearby[:5]  # Limit to first 5 mentions

        # Timestamp-based proximity
        nearby = []
        for seg in segments:
            seg_mid = (seg.start + seg.end) / 2
            if abs(seg_mid - detection.timestamp) <= window_seconds:
                nearby.append(seg)

        return nearby

    def print_detection_summary(
        self, detections: List[NameDetection], mappings: Dict[str, SpeakerNameMapping]
    ) -> None:
        """
        Print summary of name detections and mappings.

        Args:
            detections: All detected names
            mappings: Final speaker-to-name mappings
        """
        if not detections and not mappings:
            print("\n" + "=" * 60)
            print("No speaker names detected")
            print("=" * 60)
            return

        print("\n" + "=" * 60)
        print(f"Speaker Name Detection Summary")
        print("=" * 60)

        if detections:
            # Count by source
            by_source = defaultdict(list)
            for d in detections:
                by_source[d.source].append(d)

            print(f"\nDetected {len(detections)} name mentions from {len(by_source)} sources:")
            for source, dets in by_source.items():
                unique_names = set(d.name for d in dets)
                print(f"  - {source}: {len(unique_names)} unique names")

        if mappings:
            print(f"\nSpeaker Mappings ({len(mappings)} speakers):")
            print("-" * 60)
            for speaker_label, mapping in sorted(mappings.items()):
                sources = set(d.source for d in mapping.detections)
                print(
                    f"  {speaker_label} → {mapping.name} "
                    f"(confidence: {mapping.confidence:.0%}, "
                    f"sources: {', '.join(sources)})"
                )

        print("=" * 60)
