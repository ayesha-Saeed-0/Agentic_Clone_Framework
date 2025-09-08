import os
import torch.serialization
import streamlit as st
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from pydub import AudioSegment
import google.generativeai as genai
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import whisper
# ============================
# 🔹 Setup Gemini API
# ============================
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")

# ============================
# 🔹 TTS model setup
# ============================
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])

@st.cache_resource
def load_tts_model():
    return TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True
    )

tts = load_tts_model()

# ============================
# 🔹 Global variables for sharing data between tools
# ============================
current_speaker_wav = None
current_source_audio = None

# ============================
# 🔹 Utility Functions
# ============================
def split_text(text, max_chars=200):
    sentences, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 < max_chars:
            current += " " + word
        else:
            sentences.append(current.strip())
            current = word
    if current:
        sentences.append(current.strip())
    return sentences

def text_to_speech(text, speaker_wav=None, output_dir="cloned_chunks"):
    if not speaker_wav:
        raise ValueError("A speaker voice sample (speaker_wav) is required for text-to-speech with XTTS.")
    os.makedirs(output_dir, exist_ok=True)
    all_files = []
    sentences = split_text(text)
    for j, sent in enumerate(sentences, 1):
        if not sent.strip():
            continue
        out_file = f"{output_dir}/chunk_{j:02d}.wav"
        tts.tts_to_file(
            text=sent,
            file_path=out_file,
            speaker_wav=speaker_wav,
            language="en"
        )
        all_files.append(out_file)

    combined = AudioSegment.empty()
    for file in all_files:
        combined += AudioSegment.from_wav(file)

    final_output = f"{output_dir}/final_output.wav"
    combined.export(final_output, format="wav")
    return final_output

def transcribe_audio(audio_file):
    """Transcribe audio locally using Whisper (no API key required)"""
    model = whisper.load_model("base")  # models: tiny, base, small, medium, large
    result = model.transcribe(audio_file)
    return {"text": result["text"], "segments": result["segments"]}

# ============================
# 🔹 Enhanced Tool Functions (fully agentic)
# ============================

def generate_content_tool(query):
    """Generate text content on any topic requested by the user"""
    try:
        prompt = f"Generate engaging content about: {query}. Keep it informative and interesting, under 150 words."
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating content: {str(e)}"

def text_to_speech_tool(text):
    """Convert any text to speech using the cloned voice"""
    global current_speaker_wav
    if not current_speaker_wav:
        return "Error: No speaker voice sample available. User needs to upload a voice sample first."
    try:
        output_file = text_to_speech(text, speaker_wav=current_speaker_wav)
        return f"Successfully converted text to speech. Audio file created at: {output_file}"
    except Exception as e:
        return f"Error in text-to-speech conversion: {str(e)}"

def speech_to_speech_tool(input_description):
    """Convert speech from uploaded audio to the cloned target voice"""
    global current_speaker_wav, current_source_audio
    if not current_speaker_wav:
        return "Error: No speaker voice sample available. User needs to upload a voice sample first."
    if not current_source_audio:
        return "Error: No source audio available. User needs to upload source audio first."
    try:
        # Transcribe the source audio
        result = transcribe_audio(current_source_audio)
        transcribed_text = result["text"]
        
        # Convert transcribed text to cloned voice
        output_file = text_to_speech(transcribed_text, speaker_wav=current_speaker_wav)
        return f"Successfully converted source speech to cloned voice. Original text: '{transcribed_text}'. Audio file created at: {output_file}"
    except Exception as e:
        return f"Error in speech-to-speech conversion: {str(e)}"

def transcribe_audio_tool(input_description):
    """Transcribe uploaded audio to text"""
    global current_source_audio
    if not current_source_audio:
        return "Error: No source audio available. User needs to upload source audio first."
    try:
        result = transcribe_audio(current_source_audio)
        return f"Transcription: {result['text']}"
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"



# ============================
# 🔹 LangChain Agent Tools (Enhanced)
# ============================
tools = [
    Tool(
        name="GenerateContent",
        func=generate_content_tool,
        description="Generate text content on any topic. Use this when the user asks for information, facts, stories, explanations, or any text content about a specific subject. Input should be the topic or query."
    ),
    Tool(
        name="TextToSpeech", 
        func=text_to_speech_tool,
        description="Convert any text to speech using a cloned voice. Use this when the user wants to hear text spoken aloud, create audio from text, or generate voice recordings. Input should be the exact text to convert to speech."
    ),
    Tool(
        name="SpeechToSpeech",
        func=speech_to_speech_tool, 
        description="Convert speech from uploaded source audio into the cloned target voice. Use this when the user wants to change the voice of existing audio or clone their voice onto different speech content. Requires both source audio and voice sample to be uploaded."
    ),
    Tool(
        name="TranscribeAudio",
        func=transcribe_audio_tool,
        description="Convert uploaded audio to text. Use this when the user wants to know what was said in an audio file or needs a text version of speech. Requires source audio to be uploaded."
    ),
   
]

# Initialize the agent with more detailed instructions
agent = initialize_agent(
    tools, 
    llm, 
    agent="zero-shot-react-description", 
    verbose=True,
    agent_kwargs={
        "prefix": """You are an intelligent voice assistant that can help users with various audio and speech tasks. You have access to several tools that allow you to:

1. Generate content on any topic
2. Convert text to speech using cloned voices  
3. Convert speech from one voice to another
4. Transcribe audio to text


Always think about what the user is asking for and use the appropriate tools. You can chain multiple tools together if needed. For example:
- If someone asks "tell me about space in my voice", first generate content about space, then convert it to speech
- If someone asks "what does my uploaded audio say in a different voice", first transcribe it, then convert the transcription to speech
- Always check system status if you encounter errors about missing files

Be helpful and creative in interpreting user requests. Don't rely on specific keywords - understand the intent behind their request."""
    }
)

# ============================
# 🔹 Streamlit UI
# ============================
st.title("🤖 Fully Agentic Voice Framework (Gemini + XTTS)")
st.write("Give me any natural language command and I'll figure out what to do!")

# Examples to help users
with st.expander("💡 Example Commands"):
    st.write("""
    **Content Generation + Voice:**
    - "Tell me an interesting fact about dolphins in my voice"
    - "Create a short story about robots and read it to me"
    - "Explain quantum physics simply and make it audio"
    
    **Voice Conversion:**
    - "Make the uploaded audio sound like my voice instead"
    - "Change the speaker in my audio file"
    
    **Transcription:**
    - "What does my uploaded audio say?"
    - "Convert my audio to text"
    
    **Combinations:**
    - "Transcribe my audio then read it back in a different tone"
    - "Tell me about cooking pasta, but make it sound dramatic"
    """)

task = st.text_area(
    "What would you like me to do?",
    placeholder="e.g., 'Create an interesting story about time travel and read it in my voice'"
)

# File uploads
col1, col2 = st.columns(2)
with col1:
    speaker_file = st.file_uploader("Upload voice sample to clone (required for speech generation)", type=["wav", "mp3"])
with col2:
    source_audio = st.file_uploader("Upload source audio (optional, for speech-to-speech)", type=["wav", "mp3"])

# Handle file uploads and set global variables
if speaker_file:
    speaker_path = f"temp_{speaker_file.name}"
    with open(speaker_path, "wb") as f:
        f.write(speaker_file.getbuffer())
    current_speaker_wav = speaker_path
    st.success("✅ Voice sample uploaded successfully!")

if source_audio:
    source_path = f"temp_{source_audio.name}"
    with open(source_path, "wb") as f:
        f.write(source_audio.getbuffer())
    current_source_audio = source_path
    st.success("✅ Source audio uploaded successfully!")

if st.button("🚀 Let the Agent Handle It", type="primary"):
    if not task.strip():
        st.error("Please enter a command.")
    else:
        try:
            with st.spinner("🤖 Agent is thinking and working..."):
                # Let the agent handle everything - no keyword matching!
                result = agent.invoke({"input": task})
            
            st.write("### 🎯 Agent Response:")
            if isinstance(result, dict) and 'output' in result:
                st.write(result['output'])
                result_text = result['output']
            else:
                st.write(result)
                result_text = str(result)
            
            # Check if audio was generated and provide playback/download
            if os.path.exists("cloned_chunks/final_output.wav"):
                st.success("🎵 Audio generated successfully!")
                
                # Play the audio
                st.audio("cloned_chunks/final_output.wav")
                
                # Download button
                with open("cloned_chunks/final_output.wav", "rb") as audio_file:
                    st.download_button(
                        label="📥 Download Generated Audio",
                        data=audio_file.read(),
                        file_name="generated_voice.wav",
                        mime="audio/wav"
                    )
            
        except Exception as e:
            st.error(f"Error: {e}")
            
            # Helpful debugging info
            with st.expander("🔧 Debug Information"):
                st.write(f"**Speaker wav available:** {current_speaker_wav is not None}")
                st.write(f"**Source audio available:** {current_source_audio is not None}")
                st.write(f"**Task:** {task}")
                if current_speaker_wav is None:
                    st.write("💡 **Tip:** Many tasks require a voice sample to be uploaded first!")

