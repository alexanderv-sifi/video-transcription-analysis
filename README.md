# Video Transcription and Analysis

A local video processing tool that combines **GPU-accelerated audio transcription** (using MLX Whisper for Apple Silicon) with **visual analysis** (using Ollama vision models) and **advanced correction features**. Process videos entirely on your local machine with no cloud dependencies.

**⚡ GPU-Accelerated**: Optimized for Apple Silicon (M1/M2/M3) with MLX - process 1-hour videos in ~25 minutes with medium model quality!

## Features

### Core Transcription
- **GPU-Accelerated Whisper**: MLX-optimized for Apple Silicon (3-4x faster than CPU)
- **Speaker Diarization**: Automatic "who spoke when" detection with MPS acceleration
- **Multiple Output Formats**: JSON, TXT, SRT subtitles

### Advanced Features ✨
- **🎯 Speaker Identification**: Train the system to recognize specific voices
- **📚 Custom Vocabulary**: Add domain-specific terms for better recognition
- **📝 Dictionary Corrections**: Automatic pattern-based error fixes
- **🤖 LLM Corrections**: Context-aware corrections via local Ollama
- **🎓 Learning System**: Learn corrections from your manual edits

### Visual Analysis
- **Frame-by-Frame Analysis**: Using Ollama vision models
- **Combined Summaries**: Integrate audio and visual insights
- **Flexible Sampling**: Configure FPS and frame limits

## Installation

### Prerequisites

1. **Python 3.10+**

2. **Ollama** (for visual analysis and LLM corrections):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull models
   ollama pull llama3.2-vision    # For visual analysis
   ollama pull llama3.2           # For LLM corrections
   ```

3. **FFmpeg** (for audio extraction):
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   ```

4. **HuggingFace Token** (for speaker diarization):
   - Create account at https://huggingface.co
   - Get token from https://huggingface.co/settings/tokens
   - Accept pyannote model conditions at https://huggingface.co/pyannote/speaker-diarization-3.1

### Install Package

```bash
# Clone repository
git clone https://github.com/alexanderv-sifi/video-transcription-analysis.git
cd video-transcription-analysis

# Install with uv (recommended) or pip
uv pip install -e .
# OR
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

## Quick Start

### Basic Transcription

```bash
# Simple transcription with speaker diarization
python examples/transcribe_only.py /path/to/video.mp4
```

Output:
```
SPEAKER_00: Welcome to the meeting. Let's discuss the AdsWizz platform.
SPEAKER_01: Thanks. I have questions about TLS 1.2 requirements...
```

## Speaker Identification

Train the system to recognize specific speakers by voice.

### 1. Enroll Speakers

```bash
# Enroll a speaker with 3-5 audio samples (10-30 seconds each)
python examples/enroll_speaker.py enroll "Alexander" \
    sample1.wav sample2.wav sample3.wav

# List enrolled speakers
python examples/enroll_speaker.py list

# View database stats
python examples/enroll_speaker.py stats
```

### 2. Use Speaker Identification

```bash
# Transcribe with speaker identification
python examples/transcribe_only.py video.mp4 --speaker-db speakers.json
```

Output:
```
Alexander: Welcome to the meeting. Let's discuss the AdsWizz platform.
Sarah: Thanks. I have questions about TLS 1.2 requirements...
```

### How It Works

1. **Diarization**: First detects who spoke when (SPEAKER_00, SPEAKER_01, etc.)
2. **Identification**: Matches each speaker against enrolled voice profiles
3. **Replacement**: Replaces generic labels with actual names

**Technology**: Uses SpeechBrain ECAPA-TDNN embeddings (1.71% EER on VoxCeleb) for state-of-the-art speaker recognition.

## Custom Vocabulary

Improve recognition of domain-specific terms, company names, and technical jargon.

### Manage Vocabulary

```bash
# Add custom terms
python examples/manage_vocabulary.py add "AdsWizz" --category companies
python examples/manage_vocabulary.py add "SFTP" --category technical_terms
python examples/manage_vocabulary.py add "TLS 1.2" --category technical_terms

# List all vocabulary
python examples/manage_vocabulary.py list

# Preview the prompt that will be built
python examples/manage_vocabulary.py preview
```

### Use Vocabulary

```bash
# Transcribe with custom vocabulary
python examples/transcribe_only.py video.mp4 --vocabulary vocabulary.yaml
```

**How It Works**: Vocabulary terms are used to build an `initial_prompt` for Whisper, which significantly improves recognition of specialized terminology.

## Dictionary Corrections

Apply automatic pattern-based corrections to fix common transcription errors.

### Manage Corrections

```bash
# Add correction rules
python examples/manage_corrections.py add "as was" "AdsWizz"
python examples/manage_corrections.py add "Ed wiz" "AdsWizz"
python examples/manage_corrections.py add "TLS 1\.2" "TLS 1.2"

# List all rules
python examples/manage_corrections.py list

# Test corrections
python examples/manage_corrections.py test "We use as was for the Ed wiz platform"
# Output: "We use AdsWizz for the AdsWizz platform"
```

### Use Corrections

```bash
# Transcribe with dictionary corrections
python examples/transcribe_only.py video.mp4 --corrections corrections.yaml
```

**How It Works**: After transcription, text is processed through regex-based pattern matching with word boundaries to avoid partial replacements.

## LLM Corrections

Use a local LLM (via Ollama) for context-aware corrections that simple patterns can't handle.

### Enable LLM Corrections

```bash
# Make sure Ollama is running with llama3.2
ollama pull llama3.2

# Transcribe with LLM corrections
python examples/transcribe_only.py video.mp4 \
    --corrections corrections.yaml \
    --enable-llm-corrections \
    --domain-context "Ad tech platform discussion about security and APIs"
```

**How It Works**:
- Runs after dictionary corrections (tier 3)
- Chunks long transcripts at speaker boundaries
- Uses low temperature (0.1) for consistency
- Preserves speaker labels and structure
- **Slow but powerful** - adds 2-5 minutes per hour of transcript

**Use When**: You need context-aware corrections for ambiguous terms, grammar fixes, or domain-specific understanding.

## Learning System

Automatically learn corrections from your manual edits.

### Learn from Edits

```bash
# 1. Transcribe a video
python examples/transcribe_only.py meeting.mp4

# 2. Manually edit the transcript (fix errors)
#    output/meeting_transcript.txt -> output/meeting_edited.txt

# 3. Learn from your edits
python examples/learn_corrections.py \
    output/meeting_transcript.txt \
    output/meeting_edited.txt

# The tool will:
# - Analyze differences between original and edited
# - Suggest new vocabulary terms
# - Suggest new dictionary rules
# - Ask you to approve each suggestion
```

### Auto-approve Mode

```bash
# Automatically accept all suggestions
python examples/learn_corrections.py \
    original.txt edited.txt \
    --auto-approve
```

**How It Works**:
- Uses word-level diffing to identify changes
- Filters noise (whitespace, capitalization)
- Groups similar patterns
- Calculates confidence scores
- Distinguishes vocabulary terms from correction rules

## Complete Workflow Example

Here's a complete end-to-end workflow combining all features:

```bash
# 1. SETUP: Enroll your voice
python examples/enroll_speaker.py enroll "Alexander" \
    voice1.wav voice2.wav voice3.wav

# 2. SETUP: Add company-specific terms
python examples/manage_vocabulary.py add "AdsWizz" --category companies
python examples/manage_vocabulary.py add "Simplifi" --category companies

# 3. INITIAL TRANSCRIPTION: Basic transcription
python examples/transcribe_only.py meeting.mp4 \
    --speaker-db speakers.json \
    --vocabulary vocabulary.yaml

# 4. CORRECTION: Manually fix remaining errors
# Edit: output/meeting_transcript.txt

# 5. LEARNING: Learn from your edits
python examples/learn_corrections.py \
    output/meeting_transcript.txt \
    output/meeting_edited.txt

# 6. FUTURE TRANSCRIPTIONS: Now fully optimized!
python examples/transcribe_only.py future_meeting.mp4 \
    --speaker-db speakers.json \
    --vocabulary vocabulary.yaml \
    --corrections corrections.yaml
```

## Four-Tier Correction System

The system applies corrections in this order:

```
1. Vocabulary (initial_prompt) → Guides Whisper during transcription
                ↓
2. Dictionary Rules → Fast pattern-based fixes
                ↓
3. LLM Corrections (optional) → Context-aware fixes
                ↓
4. Manual Learning → Improve 1 & 2 from your edits
```

**Philosophy**: Start fast and deterministic, add intelligence only where needed.

## CLI Tools Reference

### transcribe_only.py
Main transcription tool with all features.

```bash
# All features combined
python examples/transcribe_only.py video.mp4 \
    --model medium \
    --speaker-db speakers.json \
    --vocabulary vocabulary.yaml \
    --corrections corrections.yaml \
    --enable-llm-corrections \
    --domain-context "Technical discussion about ad tech"

# Without diarization
python examples/transcribe_only.py video.mp4 --no-diarization
```

### enroll_speaker.py
Manage speaker voice profiles.

```bash
# Commands
enroll_speaker.py enroll <name> <audio1> <audio2> ...
enroll_speaker.py list
enroll_speaker.py remove <name>
enroll_speaker.py info <name>
enroll_speaker.py stats
```

### manage_vocabulary.py
Manage custom vocabulary.

```bash
# Commands
manage_vocabulary.py add <term> [--category <cat>]
manage_vocabulary.py remove <term>
manage_vocabulary.py list [--category <cat>]
manage_vocabulary.py clear --category <cat>
manage_vocabulary.py preview
```

### manage_corrections.py
Manage dictionary correction rules.

```bash
# Commands
manage_corrections.py add <pattern> <replacement> [--case-sensitive]
manage_corrections.py remove <pattern>
manage_corrections.py list
manage_corrections.py clear
manage_corrections.py test <text>
```

### learn_corrections.py
Learn from manual edits.

```bash
# Commands
learn_corrections.py <original.txt> <edited.txt> [--auto-approve] [--dry-run]
```

## Whisper Model Selection (MLX on Apple Silicon)

| Model  | Size  | Time (1h video) | WER  | Quality | Recommended |
|--------|-------|-----------------|------|---------|-------------|
| tiny   | 39M   | ~8 min          | 7.5% | Basic   | ⚠️ **Testing only** |
| base   | 74M   | ~12 min         | 5.0% | Good    | Quick drafts |
| small  | 244M  | ~15 min         | 3.4% | Better  | ⭐ Balance |
| medium | 769M  | ~25 min         | 2.9% | Great   | ✅ Default |
| large  | 1550M | ~43 min         | 3.0% | Same    | ❌ Slower |
| turbo  | 809M  | ~20 min         | ~3.0% | Great   | Speed |

**Note**: Times for M1 Max with MLX. WER = Word Error Rate (lower is better).

### ⚠️ Important: Tiny Model Limitations

**DO NOT use the `tiny` model for production or real transcriptions!** The tiny model has serious quality issues:

- **Hallucination prone**: Often produces repetitive output like "then then then then..."
- **High error rate**: 7.5% WER means ~1 in 13 words is wrong
- **Poor technical terms**: Struggles with domain-specific vocabulary
- **Unreliable output**: May produce unusable transcripts

The tiny model is **ONLY** suitable for:
- Quick system testing
- Verifying your installation works
- Development/debugging

**For actual transcriptions, use at minimum the `small` model (3.4% WER, ~15 min/hour).**

## Performance Tips

### For Quality
```bash
--model medium \
--speaker-db speakers.json \
--vocabulary vocabulary.yaml \
--corrections corrections.yaml \
--enable-llm-corrections
```

### For Speed
```bash
--model small \
--corrections corrections.yaml
# Skip LLM corrections (saves 2-5 min/hour)
```

### For Testing
```bash
--model tiny
# ~8 minutes for 1-hour video
```

## Output Files

```
output/
├── video_transcript.txt          # Plain text with speaker labels
├── video_transcript.srt          # SRT subtitles
├── video_transcript.json         # Detailed JSON with segments
├── video_analysis.json           # Frame-by-frame (if using processor)
├── video_analysis.md             # Frame-by-frame markdown
└── video_combined_summary.txt    # Combined audio + visual
```

## Troubleshooting

### Speaker Diarization Not Working
- Set `HF_TOKEN` in `.env`
- Accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
- Check logs for authentication errors

### Speaker Identification Low Confidence
- Provide 3-5 diverse audio samples per speaker
- Use 10-30 second samples with clear speech
- Vary recording conditions
- Avoid background noise

### LLM Corrections Failing
- Ensure Ollama is running: `ollama list`
- Check model is pulled: `ollama pull llama3.2`
- Verify Ollama URL (default: `http://localhost:11434`)

### Learning Not Finding Patterns
- Lower `--min-occurrences` threshold
- Lower `--min-confidence` threshold
- Ensure edits are consistent (fix same error the same way)

## Architecture

```
src/video_transcription/
├── transcriber.py       # Core transcription with all integrations
├── speaker_db.py        # Speaker enrollment & identification
├── vocabulary.py        # Custom vocabulary management
├── corrector.py         # Dictionary-based corrections
├── llm_corrector.py     # LLM-based corrections
├── learning.py          # Learning from manual edits
├── analyzer.py          # Visual frame analysis
└── processor.py         # Combined audio + visual

examples/
├── transcribe_only.py           # Main CLI tool
├── enroll_speaker.py            # Speaker management
├── manage_vocabulary.py         # Vocabulary management
├── manage_corrections.py        # Corrections management
├── learn_corrections.py         # Learning from edits
├── process_video.py             # Combined processing
└── analyze_only.py              # Visual analysis only
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy src/
```

## License

MIT License

## Acknowledgments

- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - GPU-accelerated transcription
- [SpeechBrain](https://speechbrain.github.io/) - Speaker recognition
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Ollama](https://ollama.com/) - Local LLM inference
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper models
- [FFmpeg](https://ffmpeg.org/) - Media processing
