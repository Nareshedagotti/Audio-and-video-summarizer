import os
import yt_dlp
from pydub import AudioSegment
import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_groq import ChatGroq
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Device setup for Whisper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Whisper Pipeline
def init_whisper_pipeline():
    model_id = "distil-whisper/distil-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=False,
    )

# Initialize Groq LLM for summarization
def init_groq_chat(model_name):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
    )

# Download YouTube audio
def download_youtube_audio(url):
    output_file = "audio_output.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_file,
        'ffmpeg_location': r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return "audio_output.wav"

# Split audio into smaller chunks
def split_audio(file_path, chunk_duration_ms=30000):  # 30 seconds = 30,000 ms
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_file = f"{file_path.rsplit('.', 1)[0]}_chunk_{i // chunk_duration_ms}.wav"
        chunk.export(chunk_file, format="wav")
        chunks.append(chunk_file)
    return chunks

# Transcribe audio using Whisper
def transcribe_audio(file_path, whisper_pipe):
    result = whisper_pipe(file_path)
    return result["text"].strip()

# Split transcript into chunks for LLM input
def split_transcript(transcript, max_tokens=2048):
    tokens = transcript.split()
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(" ".join(tokens[i:i + max_tokens]))
    return chunks

# Summarize transcript using Groq LLM
def summarize_transcript(transcript, model_name="mixtral-8x7b-32768"):
    chat = init_groq_chat(model_name)
    prompt = f"""
    Summarize the following audio transcription in the following format:
    1. Start with "This audio is about...".
    2. List the key points discussed in the transcription.
    3. Make it concise and structured.

    Transcription:
    {transcript}
    """
    response = chat.invoke(prompt)
    return response.content.strip()  # Access content from AIMessage

# Process audio and summarize
def process_audio_and_summarize(file_path, model_name):
    try:
        whisper_pipe = init_whisper_pipeline()

        # Step 1: Split audio into manageable chunks
        audio_chunks = split_audio(file_path, chunk_duration_ms=30000)  # 30-second chunks

        # Step 2: Transcribe each chunk and combine transcripts
        full_transcript = ""
        for chunk in audio_chunks:
            print(f"Transcribing chunk: {chunk}")
            full_transcript += transcribe_audio(chunk, whisper_pipe) + "\n"
            os.remove(chunk)  # Clean up each chunk after processing

        # Return full transcript for display before summarization
        print("Full transcript generated. Displaying it first.")
        transcript_chunks = split_transcript(full_transcript, max_tokens=2048)

        # Step 3: Summarize the transcript in chunks (if needed)
        summary = ""
        for chunk in transcript_chunks:
            summary += summarize_transcript(chunk, model_name) + "\n"

        return summary, full_transcript
    except Exception as e:
        return f"Error: {str(e)}", None

# Process input for transcription and summarization
def process_input(input_type, youtube_url, uploaded_file, model_name):
    try:
        # Process input: Download YouTube audio or use uploaded file
        if input_type == "YouTube URL":
            audio_file = download_youtube_audio(youtube_url)
        elif input_type in ["Audio File", "Video File"]:
            audio_file = uploaded_file.name
        else:
            return "Invalid input type selected.", None

        # Transcribe and summarize the audio
        summary, transcript = process_audio_and_summarize(audio_file, model_name)

        # Clean up the downloaded or uploaded audio file
        if input_type == "YouTube URL":
            os.remove(audio_file)

        return summary, transcript
    except Exception as e:
        return f"Error: {str(e)}", None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# YouTube/Audio/Video Summarization using Whisper and Groq")
    
    # Row 1: Input and Summarization Output
    with gr.Row():
        # Left Column: Input Type, YouTube URL, File Upload, and Model Selection
        with gr.Column(scale=1):
            input_type = gr.Radio(
                label="Input Type",
                choices=["YouTube URL", "Audio File", "Video File"],
                value="YouTube URL",
            )
            youtube_url = gr.Textbox(
                label="YouTube URL",
                placeholder="Enter the YouTube video URL here...",
                visible=True,
            )
            uploaded_file = gr.File(label="Upload Audio/Video File", visible=False)
            model_name = gr.Radio(
                label="Select Groq LLM Model",
                choices=["mixtral-8x7b-32768", "llama2-70b-4096"],
                value="mixtral-8x7b-32768",
            )
        # Right Column: Summarized Text Output
        with gr.Column(scale=1):
            summary_output = gr.Textbox(
                label="Summarized Text", interactive=False, lines=10
            )
    
    # Row 2: Full Transcript
    with gr.Row():
        with gr.Column():
            transcript_output = gr.Textbox(
                label="Full Transcript", interactive=False, lines=20
            )

    # Button to trigger processing
    submit_button = gr.Button("Process Input")

    # Show/Hide inputs based on input type
    def update_input_visibility(input_type):
        return gr.update(visible=input_type == "YouTube URL"), gr.update(
            visible=input_type in ["Audio File", "Video File"]
        )

    input_type.change(
        update_input_visibility,
        inputs=input_type,
        outputs=[youtube_url, uploaded_file],
    )

    submit_button.click(
        process_input,
        inputs=[input_type, youtube_url, uploaded_file, model_name],
        outputs=[summary_output, transcript_output],
    )

# Launch the Gradio app
demo.launch(share=True)