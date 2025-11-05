# Video Transcription and Analysis

A local video processing tool that combines GPU-accelerated audio transcription (using MLX Whisper for Apple Silicon) with visual analysis (using Ollama vision models). Process videos entirely on your local machine with no cloud dependencies.

**⚡ GPU-Accelerated**: Optimized for Apple Silicon (M1/M2/M3) with MLX - process 1-hour videos in ~25 minutes with medium model quality!

## Features

- **Audio Transcription**: Transcribe video audio using Whisper models
- **Visual Analysis**: Analyze video frames using Ollama vision models (e.g., llama3.2-vision)
- **Combined Processing**: Get insights from both audio and visual content
- **Multiple Output Formats**: JSON, TXT, SRT, Markdown
- **Fully Local**: All processing happens on your machine

## Prerequisites

1. **Python 3.10+**

2. **Ollama**: Install and run Ollama locally
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull a vision model
   ollama pull llama3.2-vision

   # Verify Ollama is running
   ollama list
   ```

3. **FFmpeg**: Required for audio extraction
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## Installation

1. Clone or navigate to the repository:
   ```bash
   cd video-transcription-analysis
   ```

2. Install with uv (recommended) or pip:
   ```bash
   # Using uv
   uv pip install -e .

   # Or using pip
   pip install -e .
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

## Quick Start

### Basic Example

```python
from pathlib import Path
from video_transcription import VideoProcessor

# Initialize processor (MLX auto-uses GPU on Apple Silicon)
processor = VideoProcessor(
    whisper_model="medium",  # Options: tiny, base, small, medium, large, turbo
    ollama_model="llama3.2-vision"
)

# Process a video
video_path = Path("your_video.mp4")
result = processor.process_video(
    video_path,
    analysis_fps=1.0,  # Extract 1 frame per second
    max_frames=10      # Limit to 10 frames for testing
)

# Save results
output_dir = Path("output")
processor.save_results(result, output_dir, "my_video")

# Access results
print("Transcription:", result.transcription.full_text)
print("Summary:", result.combined_summary)
```

### Command-Line Usage

```bash
# Run the example script
python examples/process_video.py /path/to/video.mp4

# With custom settings
python examples/process_video.py \
    /path/to/video.mp4 \
    --fps 0.5 \
    --max-frames 20 \
    --whisper-model medium \
    --output-dir ./my_output
```

## Usage Guide

### Transcription Only

```python
from pathlib import Path
from video_transcription import VideoTranscriber

# MLX auto-uses GPU on Apple Silicon
transcriber = VideoTranscriber(model_name="medium")
result = transcriber.transcribe(Path("video.mp4"))

# Save in different formats
transcriber.save_transcription(result, Path("output.txt"), format="txt")
transcriber.save_transcription(result, Path("output.srt"), format="srt")
transcriber.save_transcription(result, Path("output.json"), format="json")
```

### Visual Analysis Only

```python
from pathlib import Path
from video_transcription import VideoAnalyzer

analyzer = VideoAnalyzer(
    ollama_url="http://localhost:11434",
    model="llama3.2-vision"
)

result = analyzer.analyze_video(
    Path("video.mp4"),
    fps=1.0,
    frame_prompt="Describe this scene in detail, noting any text, objects, and actions."
)

# Save analysis
analyzer.save_analysis(result, Path("analysis.json"), format="json")
analyzer.save_analysis(result, Path("analysis.md"), format="md")
```

### Combined Processing

```python
from pathlib import Path
from video_transcription import VideoProcessor

processor = VideoProcessor()

result = processor.process_video(
    Path("video.mp4"),
    analysis_fps=2.0,           # 2 frames per second
    max_frames=None,            # No limit
    generate_combined_summary=True
)

# Access different parts
print("Audio:", result.transcription.full_text)
print("Visual summary:", result.analysis.summary)
print("Combined:", result.combined_summary)

# Iterate through transcription segments
for segment in result.transcription.segments:
    print(f"{segment.start:.2f}s: {segment.text}")

# Iterate through frame analyses
for frame in result.analysis.frames:
    print(f"Frame {frame.frame_number} at {frame.timestamp:.2f}s:")
    print(f"  {frame.description}")
```

## Configuration

### Environment Variables

Create a `.env` file:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision

# Whisper Configuration (MLX-accelerated)
# MLX automatically uses GPU acceleration on Apple Silicon (M1/M2/M3)
WHISPER_MODEL=medium  # Options: tiny, base, small, medium, large, turbo

# Output Configuration
OUTPUT_DIR=./output
FRAME_EXTRACTION_FPS=1
```

### Whisper Model Selection (MLX on Apple Silicon)

| Model  | Size  | Time (1-hour video) | WER  | Quality | Recommended |
|--------|-------|---------------------|------|---------|-------------|
| tiny   | 39M   | ~8 min              | 7.5% | Basic   | Testing only |
| base   | 74M   | ~12 min             | 5.0% | Good    | Quick drafts |
| small  | 244M  | ~15 min             | 3.4% | Better  | ⭐ Best balance |
| medium | 769M  | ~25 min             | 2.9% | Great   | ✅ Default |
| large  | 1550M | ~43 min             | 3.0% | Same as medium | ❌ Slower, no benefit |
| turbo  | 809M  | ~20 min             | ~3.0% | Great   | Speed optimized |

**Note**: Times are estimated for M1 Max with MLX acceleration. WER = Word Error Rate (lower is better).

### Ollama Models

Any Ollama vision model will work. Popular options:
- `llama3.2-vision` (recommended)
- `llava`
- `bakllava`

Install models with:
```bash
ollama pull llama3.2-vision
```

## Output Files

When using `VideoProcessor.save_results()`, the following files are created:

```
output/
├── video_transcript.txt          # Plain text transcription
├── video_transcript.srt          # SRT subtitle format
├── video_transcript.json         # Detailed JSON with segments
├── video_analysis.json           # Frame-by-frame analysis (JSON)
├── video_analysis.md             # Frame-by-frame analysis (Markdown)
└── video_combined_summary.txt    # Combined audio + visual summary
```

## Performance Tips

1. **Start Small**: Use `max_frames=5` for testing before processing entire videos

2. **Adjust FPS**: Lower FPS (e.g., 0.5) for longer videos to reduce processing time

3. **Choose Right Models**:
   - For quick tests: `whisper_model="tiny"`, `analysis_fps=0.5`
   - For quality: `whisper_model="medium"`, `analysis_fps=2.0`
   - For best balance: `whisper_model="small"` (3-4x faster than medium, similar quality)

4. **GPU Acceleration**:
   - **Apple Silicon (M1/M2/M3)**: MLX automatically uses GPU - no configuration needed! ⚡
   - **Performance**: ~25 min for 1-hour video with medium model (vs 90-120 min on CPU)
   - Diarization also GPU-accelerated via MPS (Metal Performance Shaders)

5. **Batch Processing**: Process multiple videos in parallel using Python's multiprocessing

## Troubleshooting

### "Failed to connect to Ollama"
- Ensure Ollama is running: `ollama list`
- Check the URL in `.env` matches your setup
- Default is `http://localhost:11434`

### "Failed to extract audio"
- Install FFmpeg: `brew install ffmpeg` (macOS)
- Verify: `ffmpeg -version`

### Slow processing
- Use smaller Whisper model (`tiny` or `base`)
- Reduce `analysis_fps` (try 0.5 or 0.25)
- Limit frames with `max_frames`

### Out of memory
- Use smaller Whisper model
- Reduce `analysis_fps`
- Process shorter video segments

## Examples

See the `examples/` directory for complete examples:
- `process_video.py`: Full command-line processor
- `transcribe_only.py`: Audio transcription only
- `analyze_only.py`: Visual analysis only

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy src/
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for GPU-accelerated transcription on Apple Silicon
- [OpenAI Whisper](https://github.com/openai/whisper) for the original Whisper models
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Ollama](https://ollama.com/) for local LLM inference
- [FFmpeg](https://ffmpeg.org/) for media processing
