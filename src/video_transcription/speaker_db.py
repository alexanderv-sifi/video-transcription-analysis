"""
Speaker identification using voice embeddings.

This module provides speaker enrollment and identification capabilities using
SpeechBrain's ECAPA-TDNN model for state-of-the-art speaker recognition.

Architecture:
- SpeakerDatabase: Main interface for speaker management
- Stores embeddings in JSON format for simplicity and portability
- Uses cosine similarity for speaker matching
- Graceful error handling with explicit exceptions

Example:
    >>> from pathlib import Path
    >>> db = SpeakerDatabase(Path("speakers.json"))
    >>> db.enroll_speaker("Alexander", [Path("sample1.wav"), Path("sample2.wav")])
    >>> name, confidence = db.identify_speaker(Path("unknown.wav"))
    >>> print(f"Identified: {name} (confidence: {confidence:.2f})")
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)


class SpeakerDatabaseError(Exception):
    """Base exception for speaker database errors."""
    pass


class SpeakerEnrollmentError(SpeakerDatabaseError):
    """Raised when speaker enrollment fails."""
    pass


class SpeakerIdentificationError(SpeakerDatabaseError):
    """Raised when speaker identification fails."""
    pass


@dataclass
class SpeakerProfile:
    """Speaker profile with voice embedding and metadata."""

    name: str
    embedding: List[float]
    enrolled_date: str
    sample_count: int
    last_seen: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerProfile":
        """Create from dictionary."""
        return cls(**data)


class SpeakerDatabase:
    """
    Manage speaker voice profiles using SpeechBrain ECAPA-TDNN embeddings.

    This class provides speaker enrollment (registration) and identification
    (matching against known speakers). It uses state-of-the-art ECAPA-TDNN
    embeddings from SpeechBrain for high accuracy (1.71% EER on VoxCeleb).

    Attributes:
        db_path: Path to JSON database file
        threshold: Cosine distance threshold for identification (lower = stricter)
        model: SpeechBrain encoder model for embedding extraction
        speakers: Dictionary of speaker profiles

    Design decisions:
        - JSON storage for simplicity and human-readability
        - Cosine distance for similarity (standard in speaker recognition)
        - Lazy model loading (only loads when needed)
        - MPS (Apple Silicon GPU) support for faster embedding extraction
    """

    DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    DEFAULT_THRESHOLD = 0.25  # Recommended for high accuracy
    EMBEDDING_DIM = 192  # ECAPA-TDNN embedding dimension

    def __init__(
        self,
        db_path: Path,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        """
        Initialize speaker database.

        Args:
            db_path: Path to database JSON file
            threshold: Cosine distance threshold (lower = stricter matching)
            model_name: SpeechBrain model identifier
            device: Device for model ("mps", "cuda", "cpu" or None for auto)

        Raises:
            SpeakerDatabaseError: If database file is corrupted
        """
        self.db_path = db_path
        self.threshold = threshold
        self.model_name = model_name
        self._model: Optional[EncoderClassifier] = None
        self.speakers: Dict[str, SpeakerProfile] = {}

        # Auto-detect device (prefer MPS on Apple Silicon)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load existing database
        self._load_database()

        logger.info(
            f"Initialized SpeakerDatabase: {len(self.speakers)} speakers, "
            f"threshold={threshold}, device={self.device}"
        )

    @property
    def model(self) -> EncoderClassifier:
        """
        Lazy-load the embedding model.

        This defers model loading until actually needed, saving memory
        and startup time when speaker identification is not used.

        Returns:
            Loaded SpeechBrain encoder model

        Raises:
            SpeakerDatabaseError: If model fails to load
        """
        if self._model is None:
            try:
                logger.info(f"Loading SpeechBrain model: {self.model_name}")
                self._model = EncoderClassifier.from_hparams(
                    source=self.model_name,
                    savedir=f"pretrained_models/{self.model_name.split('/')[-1]}",
                    run_opts={"device": self.device}
                )
                logger.info("✓ SpeechBrain model loaded successfully")
            except Exception as e:
                raise SpeakerDatabaseError(
                    f"Failed to load SpeechBrain model '{self.model_name}': {e}"
                ) from e
        return self._model

    def enroll_speaker(
        self,
        name: str,
        audio_samples: List[Path],
        force: bool = False
    ) -> SpeakerProfile:
        """
        Enroll a new speaker with voice samples.

        Multiple audio samples are averaged to create a robust embedding
        that works across different recording conditions.

        Args:
            name: Speaker name/identifier
            audio_samples: List of audio file paths (3-5 recommended)
            force: If True, overwrite existing speaker

        Returns:
            Created speaker profile

        Raises:
            SpeakerEnrollmentError: If enrollment fails
            ValueError: If name already exists and force=False

        Best practices:
            - Provide 3-5 audio samples per speaker
            - Each sample should be 10-30 seconds of clear speech
            - Vary recording conditions (different times, environments)
            - Ensure good audio quality (16kHz+, minimal noise)
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Speaker name cannot be empty")

        if name in self.speakers and not force:
            raise ValueError(
                f"Speaker '{name}' already exists. Use force=True to overwrite."
            )

        if not audio_samples:
            raise ValueError("At least one audio sample is required")

        for audio_path in audio_samples:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract embeddings from all samples
        try:
            logger.info(f"Enrolling speaker '{name}' with {len(audio_samples)} samples...")
            embeddings = []

            for i, audio_path in enumerate(audio_samples, 1):
                logger.debug(f"  Processing sample {i}/{len(audio_samples)}: {audio_path.name}")
                emb = self._extract_embedding(audio_path)
                embeddings.append(emb)

            # Average embeddings for robustness
            avg_embedding = np.mean(embeddings, axis=0)

            # Normalize (important for cosine similarity)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            # Create profile
            profile = SpeakerProfile(
                name=name,
                embedding=avg_embedding.tolist(),
                enrolled_date=datetime.now().isoformat(),
                sample_count=len(audio_samples)
            )

            self.speakers[name] = profile
            self._save_database()

            logger.info(f"✓ Speaker '{name}' enrolled successfully")
            return profile

        except Exception as e:
            raise SpeakerEnrollmentError(
                f"Failed to enroll speaker '{name}': {e}"
            ) from e

    def identify_speaker(
        self,
        audio_path: Path,
        threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float]:
        """
        Identify speaker from audio file.

        Args:
            audio_path: Path to audio file
            threshold: Custom threshold (uses default if None)

        Returns:
            Tuple of (speaker_name, confidence) where confidence is 1-distance.
            Returns (None, distance) if no match found.

        Raises:
            SpeakerIdentificationError: If identification process fails
            FileNotFoundError: If audio file doesn't exist

        Note:
            Lower distance = more similar
            Threshold of 0.25 means 75% similarity required
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not self.speakers:
            logger.warning("No speakers enrolled in database")
            return None, 1.0

        threshold = threshold or self.threshold

        try:
            # Extract embedding from unknown audio
            unknown_emb = self._extract_embedding(audio_path)

            # Find best match
            best_match = None
            best_distance = float('inf')

            for name, profile in self.speakers.items():
                stored_emb = np.array(profile.embedding)
                distance = cosine(unknown_emb, stored_emb)

                logger.debug(f"  {name}: distance={distance:.3f}")

                if distance < best_distance:
                    best_distance = distance
                    best_match = name

            # Check if best match exceeds threshold
            if best_distance < threshold:
                confidence = 1 - best_distance
                logger.info(f"✓ Identified speaker: {best_match} (confidence: {confidence:.2f})")

                # Update last_seen
                self.speakers[best_match].last_seen = datetime.now().isoformat()
                self._save_database()

                return best_match, confidence
            else:
                logger.info(
                    f"No match found (best: {best_match} with distance {best_distance:.3f}, "
                    f"threshold: {threshold:.3f})"
                )
                return None, best_distance

        except Exception as e:
            raise SpeakerIdentificationError(
                f"Failed to identify speaker from '{audio_path}': {e}"
            ) from e

    def remove_speaker(self, name: str) -> bool:
        """
        Remove a speaker from the database.

        Args:
            name: Speaker name to remove

        Returns:
            True if speaker was removed, False if not found
        """
        if name in self.speakers:
            del self.speakers[name]
            self._save_database()
            logger.info(f"✓ Removed speaker: {name}")
            return True
        else:
            logger.warning(f"Speaker not found: {name}")
            return False

    def list_speakers(self) -> List[SpeakerProfile]:
        """Get list of all enrolled speakers."""
        return list(self.speakers.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_speakers": len(self.speakers),
            "model": self.model_name,
            "threshold": self.threshold,
            "device": self.device,
            "embedding_dim": self.EMBEDDING_DIM
        }

    def _extract_embedding(self, audio_path: Path) -> np.ndarray:
        """
        Extract embedding from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Normalized embedding vector

        Raises:
            Exception: If embedding extraction fails
        """
        # Load audio with torchaudio (speechbrain expects tensor input)
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Resample to 16kHz if needed (ECAPA-TDNN is trained on 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Encode audio to embedding
        emb = self.model.encode_batch(waveform)

        # Convert to numpy and squeeze batch dimension
        emb_np = emb.squeeze().cpu().numpy()

        # Normalize
        emb_np = emb_np / np.linalg.norm(emb_np)

        return emb_np

    def _load_database(self) -> None:
        """Load speaker database from JSON file."""
        if not self.db_path.exists():
            logger.info(f"No existing database found at {self.db_path}")
            self.speakers = {}
            return

        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)

            # Validate format
            if "speakers" not in data:
                raise ValueError("Invalid database format: missing 'speakers' key")

            # Load speakers
            self.speakers = {
                name: SpeakerProfile.from_dict(profile_data)
                for name, profile_data in data["speakers"].items()
            }

            logger.info(f"Loaded {len(self.speakers)} speakers from {self.db_path}")

        except json.JSONDecodeError as e:
            raise SpeakerDatabaseError(
                f"Corrupted database file at {self.db_path}: {e}"
            ) from e
        except Exception as e:
            raise SpeakerDatabaseError(
                f"Failed to load database from {self.db_path}: {e}"
            ) from e

    def _save_database(self) -> None:
        """Save speaker database to JSON file."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data
            data = {
                "speakers": {
                    name: profile.to_dict()
                    for name, profile in self.speakers.items()
                },
                "metadata": {
                    "model": self.model_name,
                    "threshold": self.threshold,
                    "embedding_dim": self.EMBEDDING_DIM,
                    "last_updated": datetime.now().isoformat()
                }
            }

            # Write atomically (write to temp, then rename)
            temp_path = self.db_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self.db_path)
            logger.debug(f"Saved database to {self.db_path}")

        except Exception as e:
            raise SpeakerDatabaseError(
                f"Failed to save database to {self.db_path}: {e}"
            ) from e
