# Justfile for video-transcription-analysis

# Install dependencies
install:
    uv pip install -e .

# Install with dev dependencies
install-dev:
    uv pip install -e ".[dev]"

# Run example with test video
example VIDEO:
    python examples/process_video.py {{VIDEO}} --max-frames 5

# Transcribe only
transcribe VIDEO:
    python examples/transcribe_only.py {{VIDEO}}

# Analyze only
analyze VIDEO:
    python examples/analyze_only.py {{VIDEO}} --max-frames 5

# Format code
format:
    ruff format .

# Lint code
lint:
    ruff check .

# Fix linting issues
fix:
    ruff check --fix .

# Type check
typecheck:
    mypy src/

# Run all checks
check: format lint typecheck

# Clean output directory
clean:
    rm -rf output/*
    touch output/.gitkeep

# Check Ollama connection
check-ollama:
    @echo "Checking Ollama connection..."
    @curl -s http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama is running" || echo "✗ Ollama is not running"

# List available Ollama models
ollama-models:
    ollama list
