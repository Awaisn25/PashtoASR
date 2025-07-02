import os
import base64
import json
import time
from typing import Dict
import numpy as np

from fastapi import FastAPI, Request
# from pyannote.audio import Model
from pyctcdecode import build_ctcdecoder
import librosa
import io
from pydub import AudioSegment
import uvicorn
import io
import soundfile as sf
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import wave
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoProcessor, Wav2Vec2ForCTC, \
    Wav2Vec2ProcessorWithLM, WhisperProcessor, WhisperForConditionalGeneration
import gc
import torch
import uuid as U
import librosa
from faster_whisper import WhisperModel
from starlette.responses import JSONResponse
from predict import saluz

############  models code
os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = FastAPI()
SAMPLE_RATE=16000
REQUEST_COUNT = 10
buffer_dict : Dict[str, np.array] = {}
uuid_requests_count : Dict[str, int] = {}
lang_dict : Dict[str, str] = {}
buffer = io.BytesIO()
bufferAll = io.BytesIO()

pipe_ps = ''
modelUrdu=''
processor_tiny=''
modelUrdu = ''
model_tiny=''

print('Loading Model and LM...')
model_id_ps = './mms_ps'
print('Loading Urdu Model...') 
print('Urdu model loaded...')

print('Loading VAD...')
vad, utils = torch.hub.load(repo_or_dir='./VAD/snakers4',
                              model='silero_vad',
                              source='Local')
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils
vad=vad.to('cpu')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
favicon_path = "favicon.ico"
print('running app')


@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("main_prev_optim2.html", {"request": request})


@app.post("/process_lang/")
async def process_lang_endpoint(request: Request):
    global lang_dict
    body = await request.body()
    body = json.loads(body)
    language = body['lang']
    uuid = body['uuid']

    lang_dict[uuid] = language.lower()
    print(f"Language for {uuid} set to: {language}")
    return JSONResponse(content={"message": f"Language set to {language} for {uuid}"},status_code=200)

def prog(prog:float):
    print(f'VAD Progress: {round(prog,2)}', end='\r')

@app.post("/process_file/")
async def process_file(request: Request):
    
    body = await request.body()
    body  = json.loads(body)
    wav_bytes = body['file']
    wav_bytes = base64.b64decode(wav_bytes)
    uuid = body['uuid']
    temp_lang = lang_dict[uuid]
    data, sr = librosa.load(io.BytesIO(wav_bytes), sr=16000)

    return StreamingResponse(
        generate(data, temp_lang, 'process_file'),
        media_type="text/plain",
        headers={"Content-type": "text/plain"},
    )

@app.post("/procFile/")
async def procFile(request:Request):
    body = await request.body()
    data = json.loads(body)
    link = data['link']
    uuid = data['uuid']
    temp_lang = lang_dict[uuid]
    print(link)
    start = time.time()
    resp = requests.get(link, verify=False)
    end = time.time()
    data, sr = librosa.load(io.BytesIO(resp.content), sr=SAMPLE_RATE)
    
    print(f"Fetched file in {round(end-start, 2)} seconds.")
    return StreamingResponse(
            generate(data, temp_lang, 'procFile'),
            media_type="text/plain",
            headers={"Content-type": "text/plain"},
    )


def isSpeech(wav_arr:np.ndarray):
    global vad
    audiodata = torch.from_numpy(wav_arr).float()
    speech_timestamps = get_speech_timestamps(
        audiodata, vad, sampling_rate=SAMPLE_RATE,
        threshold=0.25, min_speech_duration_ms=150,
        min_silence_duration_ms=200
    )
    return len(speech_timestamps) > 0


def generate(wav_data, lang, method):
    '''
    Used for file processing.
    '''

    print(f'File.', lang)
    full_transcription = ''
    # todo, this VAD runs only on CPU.
    (get_speech_timestamps, _, read_audio, *_) = utils
    # wav = wav.to('cuda')
    t1_vad = time.time()
    speech_timestamps = get_speech_timestamps(
        wav_data, vad, 
        sampling_rate=16000, 
        threshold=0.25,
        max_speech_duration_s=20,
        min_speech_duration_ms=400,
        window_size_samples=1024,
        min_silence_duration_ms=250,
        progress_tracking_callback=prog
    )
    t2_vad = time.time()
    print(f'Length of Speech Segment Array: {len(speech_timestamps)}')
    print(f'Time Taken by VAD: {round(t2_vad - t1_vad, 2)} seconds.')
    for seg in speech_timestamps:
        data = wav_data[seg['start']: seg['end']]
        if lang == 'pashto':
            data = torch.from_numpy(data)
            transcription = saluz(data, tgt_lang='pbt', src_lang='pbt')
            translation = ''
            translation = saluz(data, tgt_lang='urd', src_lang='pbt')
            
        elif lang == 'dari':
            print('Skipping Dari.')
        else:
            data = torch.from_numpy(data)
            transcription = saluz(data, tgt_lang='pbt', src_lang='urd')
        transcription = milliseconds_to_time(seg) + '    '  +  transcription + '\n'
        if lang != 'urdu':
            full_translation = milliseconds_to_time(seg) + '    ' + translation + '\n'
        # full_transcription = translation + '\n'
        if lang == 'pashto':
            ret = {'transcription':transcription, 'translation':full_translation}
        else:
            ret = {'transcription':transcription}

        print(ret)
        yield json.dumps(ret) + "\n"

def milliseconds_to_time(milliseconds):
    seconds = milliseconds['start'] / 16000
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    from_time = f'{str(int(hours)).zfill(2)}:{int(minutes):02}:{seconds:05.2f}'

    seconds = milliseconds['end'] / 16000
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    # to_time = f'{int(hours)}: {int(minutes)}: {seconds:.2f}     '
    to_time = f'{str(int(hours)).zfill(2)}:{int(minutes):02}:{seconds:05.2f}'
    return to_time + ' - ' + from_time


@app.post("/clear")
async def main(request: Request):
    global buffer_dict, uuid_requests_count

    # wav_bytes, uuid, end_of_sentence = process_live_request(body)
    body = await request.body()
    data = json.loads(body)
    print(data)
    uu=data["uuid"]
    if uu in buffer_dict:
        buffer_dict[uu] = np.empty([0], dtype=np.float32)
        print('cleared buffer_dict \n', buffer_dict)
    else:
        print(f'UUID {uu} was not found in buffer_dict.')
    
    if uu in uuid_requests_count:
        uuid_requests_count[uu]=0
        print(f'cleared uuid_requests_count \n', uuid_requests_count)
    else:
        print(f'UUID {uu} was not found in uuid_requests_count.')

    return JSONResponse(content={"message": "Cleared"},status_code=200)


def process_live_request(body):
    global buffer_dict, uuid_requests_count
    data = json.loads(body)
    # print(data.keys())
    wav_bytes = data['data']
    # print(wav_bytes)
    uuid = data['uuid']
    fvad = data['fvad']
    print(f'FVAD: {fvad}')
    end_of_sentence = data['end']
    # print(f"UUID received: {wav_bytes['type']}")
    wav_b64 = bytes(base64.b64decode(wav_bytes))
    temp_arr, sr = librosa.load(io.BytesIO(wav_b64), sr=SAMPLE_RATE)
    isSpeechDetected=isSpeech(temp_arr)

    if isSpeechDetected:
        print(f'VAD Output: {str(isSpeechDetected)}')
        if end_of_sentence:
            buffer_dict[uuid] = temp_arr
        else:
            if not uuid in buffer_dict:
                buffer_dict[uuid] = np.empty([0], dtype=np.float32)
                buffer_dict[uuid] = np.concatenate([buffer_dict[uuid], temp_arr], axis=0)
            else:
                if fvad:
                    buffer_dict[uuid]= temp_arr
                else:
                    buffer_dict[uuid] = np.concatenate([buffer_dict[uuid], temp_arr], axis=0)
        
            if (not uuid in uuid_requests_count):
                uuid_requests_count[uuid] = 1
            else:
                if (uuid_requests_count[uuid] < REQUEST_COUNT):
                    uuid_requests_count[uuid] += 1
        retArr = buffer_dict[uuid]
    else:
        print(f'VAD Output: {str(isSpeechDetected)}')
        return None

    return retArr, uuid, end_of_sentence, fvad


@app.post("/process_rtt_psUR/")
async def process_rtt_endpoint_ps(request: Request):
    global buffer_dict, uuid_requests_count
    body = await request.body()
    retVal = process_live_request(body)
    if retVal is None:
        return {'transcribe':''}
    else:
        data_arr, uuid, end_of_sentence, fvad = retVal

        if fvad:
            retObj = transcribe(data_arr, 'pashto', end_of_sentence, fvad)
        else:
            if (uuid in uuid_requests_count):
                if uuid_requests_count[uuid] >= REQUEST_COUNT:
                    print('Request overflow, clearing buffer...')
                    uuid_requests_count[uuid] = 0
                    buffer_dict[uuid] = np.empty([0], dtype=np.float32)

                    # sf.write('/media/linux/New Volume/Pushto_urdu_dari/YESSSS.wav', data_arr, 16000, format='wav')
                    transcription = transcribe(data_arr, 'pashto', end_of_sentence)
                    translation = translate(data_arr, 'urd')
                    print(f'Transcription: {transcription}')
                    print(f'Translation: {translation}')
                    gc.collect(); torch.cuda.empty_cache()
                    return {"transcribe": transcription, 'bufferOverflow':True, 'translation':translation}

        
        if end_of_sentence:
            uuid_requests_count[uuid] = 0
            buffer_dict[uuid] = np.empty([0], dtype=np.float32)
            del buffer_dict[uuid]
            print(buffer_dict)

        return retObj


@app.post("/process_rtt_urUR/")
async def process_rtt_endpoint_ur(request: Request):
    global buffer_dict, uuid_requests_count
    body = await request.body()
    retVal = process_live_request(body)
    if retVal is None:
        return {'transcribe':''}
    else:
        data_arr, uuid, end_of_sentence, fvad = retVal
        print(fvad)

    if fvad:
        retObj = transcribe(data_arr, 'urdu', end_of_sentence, fvad)
    else:
        if (uuid in uuid_requests_count) and (not fvad):
            if uuid_requests_count[uuid] >= REQUEST_COUNT:
                print('Request overflow, clearing buffer...')
                uuid_requests_count[uuid] = 0
                buffer_dict[uuid] = np.empty([0], dtype=np.float32)

                transcription = transcribe(data_arr, 'urdu', end_of_sentence)
                gc.collect(); torch.cuda.empty_cache()
                return {"transcribe": transcription, 'bufferOverflow':True}

    if end_of_sentence:
        uuid_requests_count[uuid] = 0
        buffer_dict[uuid] = np.empty([0], dtype=np.float32)
        del buffer_dict[uuid]

    return retObj


@app.post("/process_rtt_drUR/")
async def process_rtt_endpoint_dr(request: Request):
    global buffer_dict
    print('transcribing dari sentence...')
    body = await request.body()
    wav_bytes, uuid, end_of_sentence = process_live_request(body)
    # wav_bytes = await request.body()

    transcription = transcribe(wav_bytes, 'dari')
    # translation = translate(transcription)
    if end_of_sentence:
        buffer_dict[uuid].close()
        del buffer_dict[uuid]
    # return {"transcribe": base64.b64encode(transcription.encode()).decode()}

    return {"transcribe": transcription}


def transcribe(data_arr, lang, end_of_sentence, fvad):
    if lang == 'urdu':
        data_arr = torch.from_numpy(data_arr)
        transcription = saluz(data_arr, "urd")
        transcription = str(transcription)
        retObj = {"transcribe": transcription}


    if lang == 'pashto':
        data_arr = torch.from_numpy(data_arr)
        transcription = saluz(data_arr, tgt_lang="pbt", src_lang='pbt')
        transcription = str(transcription)
        retObj = {"transcribe": transcription}

        if end_of_sentence or fvad:
            # print('translating...')
            # translation = translate(transcription, 'pbt_Arab')
            translation = saluz(data_arr, tgt_lang="urd", src_lang='pbt')
            retObj = {"transcribe": transcription,
                    "translation":translation}
        else:
            retObj = {"transcribe": transcription}

    return retObj


def translate(data_arr:np.array, tgt_lang:str):
    data_arr = torch.from_numpy(data_arr)
    output = saluz(data_arr, tgt_lang=tgt_lang, src_lang='pbt')
    output = str(output)
    return output    

if __name__ == "__main__":
    uvicorn.run(app, host="<host>", port=8000, log_level="info", 
                ssl_certfile="./certs/cert.pem", 
                ssl_keyfile="./certs/key.pem")
