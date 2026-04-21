# Agentic Voice Framework

A fully agentic voice application built with Streamlit that combines Google Gemini, Coqui XTTS v2, and OpenAI Whisper. Give it a plain-English command and the AI agent figures out which tools to use — generating content, cloning voices, transcribing audio, or chaining all three together automatically.

---

## What It Does

Instead of clicking through menus, you just type what you want:

> *"Tell me an interesting fact about dolphins in my voice"*
> *"Change the speaker in my uploaded audio to sound like me"*
> *"Transcribe my audio, then read it back dramatically"*

The LangChain agent interprets the intent and chains the right tools together without any hardcoded keyword matching.

---

## Features

| Capability | Description |
|---|---|
| **Content Generation** | Generates text on any topic via Gemini 2.0 Flash |
| **Text to Speech** | Synthesizes any text in your cloned voice (XTTS v2) |
| **Speech to Speech** | Transcribes source audio, then re-speaks it in your cloned voice |
| **Transcription** | Converts uploaded audio to text using Whisper |
| **Tool Chaining** | Agent automatically combines tools for multi-step tasks |

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit torch TTS pydub openai-whisper \
            google-generativeai langchain langchain-google-genai
```

> XTTS v2 requires a CUDA GPU for reasonable speed. Set your runtime to GPU in Colab (Runtime → Change runtime type → T4 GPU).

### 2. Set your Gemini API key

```python
# In the script or as an environment variable
GOOGLE_API_KEY="your_key_here"
```

Or export it before running:

```bash
export GOOGLE_API_KEY="your_key_here"
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## How to Use

1. **Type a command** in the text area — be as natural as you like
2. **Upload a voice sample** (WAV or MP3, 10–30 seconds) — required for any speech output
3. **Upload source audio** (optional) — needed only for speech-to-speech or transcription tasks
4. Hit **"Let the Agent Handle It"** and watch it work

### Example Commands

```
"Create a short story about robots and read it to me"
"What does my uploaded audio say?"
"Make the uploaded audio sound like my voice"
"Explain quantum physics simply and turn it into audio"
"Transcribe my audio then read it back in my voice"
```

---

## Agent Tools

The LangChain zero-shot agent has access to four tools it selects and chains automatically:

- **GenerateContent** — prompts Gemini to produce text on any topic
- **TextToSpeech** — synthesizes text in the cloned voice via XTTS v2
- **SpeechToSpeech** — transcribes source audio then re-voices it in the cloned voice
- **TranscribeAudio** — runs Whisper locally to convert audio to text

---

## Project Structure

```
├── app.py                  # Main Streamlit application
├── cloned_chunks/          # Intermediate and final WAV files (auto-created)
│   └── final_output.wav    # Merged output audio
└── temp_<filename>         # Temporary uploaded files
```

---

## Notes

- `myvoice.wav` / the uploaded voice sample is saved temporarily as `temp_<filename>` — it is not persisted between sessions
- Whisper runs locally (`base` model by default) — no API key required for transcription; swap to `"small"` or `"large"` for better accuracy
- The TTS model is cached with `@st.cache_resource` so it only loads once per session
- Audio output is playable and downloadable directly from the UI after generation
- Gemini API key should not be hardcoded — use environment variables or Streamlit secrets in production
