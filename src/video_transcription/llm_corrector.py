"""
LLM-based transcript correction using Ollama.

This module provides context-aware transcript correction using local LLMs via Ollama.
It's the third tier in the correction pipeline (after vocabulary hints and dictionary rules).

Architecture:
- LLMCorrector: Main interface for LLM-based corrections
- Chunking support for long transcripts
- Speaker label preservation
- Timeout and error handling
- Low temperature for consistency

Example:
    >>> from pathlib import Path
    >>> corrector = LLMCorrector(ollama_url="http://localhost:11434")
    >>> text = "We use as was for authentication"
    >>> corrected, stats = corrector.correct(text, domain_context="Ad tech platform")
    >>> print(corrected)
    "We use AdsWizz for authentication"
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class LLMCorrectorError(Exception):
    """Base exception for LLM corrector errors."""
    pass


class OllamaUnavailableError(LLMCorrectorError):
    """Raised when Ollama service is unavailable."""
    pass


@dataclass
class LLMCorrectionStats:
    """Statistics about LLM corrections."""

    chunks_processed: int = 0
    total_input_length: int = 0
    total_output_length: int = 0
    api_calls: int = 0
    failed_chunks: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunks_processed": self.chunks_processed,
            "total_input_length": self.total_input_length,
            "total_output_length": self.total_output_length,
            "api_calls": self.api_calls,
            "failed_chunks": self.failed_chunks,
            "errors": self.errors,
        }


class LLMCorrector:
    """
    Apply LLM-based corrections to transcripts using Ollama.

    This class uses a local LLM via Ollama to perform context-aware corrections
    that simple dictionary rules cannot handle. It preserves speaker labels and
    transcript structure while fixing transcription errors.

    Attributes:
        ollama_url: URL for Ollama API
        model: Ollama model name
        temperature: Sampling temperature (low for consistency)
        timeout: Request timeout in seconds
        chunk_size: Maximum characters per chunk

    Design decisions:
        - Low temperature (0.1) for consistent, deterministic corrections
        - Chunking at speaker boundaries to maintain context
        - Explicit prompt engineering to prevent hallucination
        - Graceful degradation when Ollama unavailable
        - Preserves speaker labels and structure
    """

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_TEMPERATURE = 0.1  # Low for consistency, not creativity
    DEFAULT_TIMEOUT = 120  # 2 minutes per chunk
    DEFAULT_CHUNK_SIZE = 2000  # Characters per chunk

    # Prompt template for corrections
    CORRECTION_PROMPT = """You are a transcript correction assistant. Your job is to fix transcription errors while preserving the exact structure and speaker labels.

CRITICAL RULES:
1. Fix ONLY clear transcription errors (mishearings, technical terms)
2. Preserve ALL speaker labels EXACTLY as they appear (e.g., SPEAKER_00:, SPEAKER_01:, etc.)
3. Do NOT reword, rephrase, or improve style - ONLY correct errors
4. Do NOT add or remove content
5. Do NOT add explanations, comments, or markdown formatting
6. Return ONLY the corrected transcript text

DOMAIN CONTEXT:
{domain_context}

TRANSCRIPT TO CORRECT:
{transcript}

CORRECTED TRANSCRIPT:"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Initialize LLM corrector.

        Args:
            ollama_url: URL for Ollama API
            model: Ollama model name (default: llama3.2)
            temperature: Sampling temperature (default: 0.1 for consistency)
            timeout: Request timeout in seconds (default: 120)
            chunk_size: Maximum characters per chunk (default: 2000)

        Raises:
            ValueError: If parameters are invalid
        """
        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")

        if chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 characters")

        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.chunk_size = chunk_size

        logger.info(
            f"Initialized LLMCorrector: model={model}, temperature={temperature}, "
            f"timeout={timeout}s, chunk_size={chunk_size}"
        )

    def correct(
        self, text: str, domain_context: Optional[str] = None
    ) -> Tuple[str, LLMCorrectionStats]:
        """
        Apply LLM-based corrections to transcript.

        Args:
            text: Transcript text to correct
            domain_context: Optional context about the domain (e.g., "Ad tech platform")

        Returns:
            Tuple of (corrected_text, statistics)

        Raises:
            OllamaUnavailableError: If Ollama service is unavailable
            LLMCorrectorError: If correction fails

        Note:
            Long transcripts are automatically chunked at speaker boundaries.
            Speaker labels (SPEAKER_XX:) are preserved throughout.
        """
        if not text or not text.strip():
            return text, LLMCorrectionStats()

        stats = LLMCorrectionStats()
        stats.total_input_length = len(text)

        # Check if Ollama is available
        if not self._check_ollama_available():
            logger.warning("Ollama unavailable - skipping LLM corrections")
            raise OllamaUnavailableError(
                f"Ollama service not available at {self.ollama_url}. "
                "Ensure Ollama is running: https://ollama.ai"
            )

        try:
            # Split into chunks at speaker boundaries
            chunks = self._chunk_text(text)
            logger.info(f"Processing {len(chunks)} chunks for LLM correction...")

            corrected_chunks = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    logger.debug(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
                    corrected = self._correct_chunk(chunk, domain_context)
                    corrected_chunks.append(corrected)
                    stats.chunks_processed += 1
                    stats.api_calls += 1

                except Exception as e:
                    logger.error(f"Failed to correct chunk {i}: {e}")
                    # Use original chunk if correction fails
                    corrected_chunks.append(chunk)
                    stats.failed_chunks += 1
                    stats.errors.append(f"Chunk {i}: {str(e)}")

            # Combine chunks
            corrected_text = "\n".join(corrected_chunks)
            stats.total_output_length = len(corrected_text)

            logger.info(
                f"✓ LLM corrections complete: {stats.chunks_processed}/{len(chunks)} "
                f"chunks successful"
            )

            return corrected_text, stats

        except OllamaUnavailableError:
            raise
        except Exception as e:
            raise LLMCorrectorError(f"LLM correction failed: {e}") from e

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks at speaker boundaries.

        This ensures each speaker's statement stays together for better
        context understanding by the LLM.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # Split by lines
        lines = text.split("\n")

        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # If adding this line exceeds chunk size and we have content, start new chunk
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _correct_chunk(self, chunk: str, domain_context: Optional[str]) -> str:
        """
        Correct a single chunk using LLM.

        Args:
            chunk: Text chunk to correct
            domain_context: Optional domain context

        Returns:
            Corrected text

        Raises:
            Exception: If API call fails
        """
        # Build prompt
        context = domain_context or "General transcription"
        prompt = self.CORRECTION_PROMPT.format(
            domain_context=context, transcript=chunk
        )

        # Call Ollama API
        corrected = self._call_ollama(prompt)

        # Clean up response (remove any markdown formatting LLM might add)
        corrected = self._clean_response(corrected)

        return corrected

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API with prompt.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response

        Raises:
            Exception: If API call fails
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.Timeout:
            raise LLMCorrectorError(
                f"Ollama request timed out after {self.timeout}s"
            )
        except requests.exceptions.ConnectionError:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.ollama_url}"
            )
        except Exception as e:
            raise LLMCorrectorError(f"Ollama API call failed: {e}") from e

    def _clean_response(self, text: str) -> str:
        """
        Clean LLM response.

        Removes markdown formatting, code blocks, or other artifacts
        the LLM might add despite instructions.

        Args:
            text: Raw LLM response

        Returns:
            Cleaned text
        """
        # Remove markdown code blocks
        text = re.sub(r"```(?:text|plaintext)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL)

        # Remove common prefixes LLMs add
        prefixes = [
            "Here is the corrected transcript:",
            "Corrected transcript:",
            "CORRECTED TRANSCRIPT:",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text.strip()

    def _check_ollama_available(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags", timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
