# Setting Up Speaker Diarization

Speaker diarization requires a HuggingFace token to download the pyannote models.

## Steps:

1. **Get a HuggingFace Token:**
   - Go to: https://huggingface.co/settings/tokens
   - Create a new token (read access is sufficient)
   - Copy the token

2. **Accept the pyannote model license:**
   - Go to: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"
   - Also visit: https://huggingface.co/pyannote/segmentation-3.0
   - Click "Agree and access repository"

3. **Set the token:**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

4. **Re-run the transcription:**
   ```bash
   source .venv/bin/activate
   python examples/transcribe_only.py ~/Movies/"2025-11-04 14-36-25.mp4" \
     --model tiny \
     --output-dir output
   ```

The transcription will now include speaker labels like SPEAKER_00, SPEAKER_01, etc.
