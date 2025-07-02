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
python app2_optim.py
```

## API Endpoints

`GET /` 
- Returns the main HTML page.
  
`POST /process_lang/`
- Set the language for a session.
- Body: `{ "lang": "<language>", "uuid": "<uuid>" }`

`POST /process_file/`
- Upload and process a base64-encoded audio file.
- Body: `{ "file": "<base64_wav>", "uuid": "<uuid>" }`

`POST /procFile/`
- Process an audio file from a URL.
- Body: `{ "link": "<file_url>", "uuid": "<uuid>" }`

`POST /process_rtt_psUR/`
- Real-time transcription and translation for Pashto.

`POST /process_rtt_urUR/`
- Real-time transcription for Urdu.

`POST /clear`
- Clear buffers for a session.
- Body: { "uuid": "<uuid>" }

## File Structure
- `app2_optim.py` - Main FastAPI application.
- `predict.py` - Contains the saluz function for model inference.
- `templates/` - HTML templates.
- `static/` - Static files (JS, CSS, etc.).
- `certs/` - SSL certificates.
- `VAD/` - VAD model files.
- `new_nllb/` - (Translation models, if used.)

## Customization
- Model Paths: Update model paths in the code as needed.
- Languages: Add or modify supported languages in the endpoints and model loading sections.
- Frontend: Update templates in the templates/ directory.

## Troubleshooting
- Ensure all model files are downloaded and paths are correct.
- Check CUDA availability for GPU inference.
- Verify SSL certificate paths if running with HTTPS.

## Acknowledgements
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Silero VAD](https://github.com/snakers4/silero-vad)
