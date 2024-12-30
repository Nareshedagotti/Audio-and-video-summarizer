# Audio and Video Summarizer

This project provides a comprehensive solution for summarizing audio and video content using state-of-the-art AI models like Whisper for transcription and Groq LLM for summarization. It supports YouTube videos, audio files, and video files, offering a user-friendly interface built with Gradio.

## Features

- **YouTube Audio Extraction**: Downloads and processes audio directly from YouTube videos.
- **Audio and Video Support**: Accepts audio or video files for summarization.
- **Transcription**: Converts speech into text using Whisper.
- **Summarization**: Generates concise and structured summaries using Groq LLM.
- **Chunking for Large Files**: Splits audio into manageable chunks for efficient processing.
- **Interactive UI**: Gradio-based interface for easy interaction.

## Requirements

- Python 3.8+
- Dependencies (install via `pip install -r requirements.txt`):
  - `yt-dlp`
  - `pydub`
  - `gradio`
  - `transformers`
  - `torch`
  - `langchain_groq`
  - `dotenv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nareshedagotti/Audio-and-video-summarizer.git
   cd Audio-and-video-summarizer
