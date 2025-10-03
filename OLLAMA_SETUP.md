# Ollama + Llama 3.1 Setup Guide

This guide shows how to set up Ollama with Llama 3.1 as a fallback for the Streamlit app when OpenAI API quota is exceeded.

## Installation

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

### 2. Start Ollama Service

```bash
ollama serve
```

This starts the Ollama API server on `http://localhost:11434`

### 3. Pull Llama 3.1 Model

**For 8B model (recommended for most systems):**
```bash
ollama pull llama3.1:8b
```

**For 70B model (requires significant RAM):**
```bash
ollama pull llama3.1:70b
```

### 4. Verify Installation

```bash
ollama list
```

You should see:
```
NAME            ID              SIZE    MODIFIED
llama3.1:8b     abc123...       4.7GB   2 hours ago
```

## Usage in Streamlit App

The app will automatically:
1. Try OpenAI API first
2. Fall back to Llama 3.1 if OpenAI quota is exceeded
3. Use static explanations if both fail

### Configuration

You can modify the model in `app_lite.py`:
```python
LLAMA_MODEL = "llama3.1:8b"  # Change to llama3.1:70b for larger model
```

### Performance Notes

- **8B model**: ~4.7GB RAM, good for most tasks
- **70B model**: ~40GB RAM, higher quality but slower
- First request may be slower as model loads into memory
- Subsequent requests are faster

## Troubleshooting

### Ollama not starting
```bash
# Check if port 11434 is in use
lsof -i :11434

# Kill existing processes
pkill -f ollama

# Restart
ollama serve
```

### Model not found
```bash
# List available models
ollama list

# Pull specific model
ollama pull llama3.1:8b
```

### Out of memory
- Use 8B model instead of 70B
- Close other applications
- Consider using cloud Ollama instance

## Cloud Alternatives

If you can't run Ollama locally, consider:
- **Groq**: Fast inference API for Llama models
- **Together AI**: Cloud inference for open models
- **Replicate**: Model hosting platform

Update the `call_llama` function to use these APIs instead of local Ollama.
