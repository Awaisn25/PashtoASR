# PashtoASR
A realtime Pashto ASR application using frontend VAD and FastAPI
---
# app2_optim.py

## Overview

`app2_optim.py` is a FastAPI-based backend service for speech-to-text transcription and translation, primarily supporting Pashto and Urdu languages. It leverages VAD (Voice Activity Detection), ASR (Automatic Speech Recognition), and translation models to process audio files and live audio streams. The service exposes several endpoints for file upload, language selection, real-time transcription, and buffer management.

## Features

- **Language Selection:** Set the language for each session/UUID.
- **Audio File Processing:** Upload and process audio files for transcription and translation.
- **Streaming & Real-Time Transcription:** Supports real-time audio streaming and transcription with buffer management.
- **Voice Activity Detection (VAD):** Detects speech segments in audio.
- **Pashto & Urdu Support:** Transcription and translation between Pashto and Urdu.
- **Buffer Management:** Handles audio buffers for each session.
- **CORS Support:** Configured for cross-origin requests.
- **Static & Template Serving:** Serves static files and HTML templates.

## Requirements

- Python 3.8+
- CUDA-enabled GPU (for model inference)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Librosa](https://librosa.org/)
- [Pydub](https://github.com/jiaaro/pydub)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)
- [requests](https://docs.python-requests.org/)
- [soundfile](https://pysoundfile.readthedocs.io/)
- [numpy](https://numpy.org/)
- [starlette](https://www.starlette.io/)
- [jinja2](https://palletsprojects.com/p/jinja/)

> **Note:** You may need to install additional dependencies for VAD and model files.

## Setup

1. **Clone the repository and navigate to the project directory.**

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download and place the required models:**
    - Pashto ASR model in `./mms_ps`
    - Urdu Whisper model in `./whisperUrdu`
    - VAD model in `./VAD/snakers4`
    - (Add instructions for other models as needed)

4. **SSL Certificates:**
    - Place your SSL certificate and key in `./certs/10.23.23.9/` as `cert.pem` and `key.pem`.

5. **Templates and Static Files:**
    - Place your HTML templates in the `templates/` directory.
    - Place static assets in the `static/` directory.

## Usage

### Running the Server

```sh
python [app2_optim.py](http://_vscodecontentref_/0)
