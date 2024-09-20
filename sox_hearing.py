import warnings

import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from speech_recognition import AudioData
import io 
import soundfile as sf 
import numpy as np


##### Send model work to this device...
#### TODO: conditionalize with torch.cuda.is_available()
if torch.cuda.is_available():
    torch_dtype = torch.float16 
    worker_device = "cuda:0"
else:
    torch_dtype = torch.float32
    worker_device="cpu"

#######################################################################################
# Name: STTInferenceProcessManager
#
# Purpose:  Encapsulates loading/managing STT (Speech-to-Text) model state as well as 
#            executing inference using insanely-fast whisper.
# 
# Caveats:  
#
#######################################################################################
class InferenceManager:

    def __init__(self):
        self.model_name = 'distil-whisper/distil-large-v2'

        self.pipe = pipeline (
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=torch_dtype,
            device=worker_device,
            #model_kwargs={"attn_implementation": "flash_attention_2"},
        )
 
 ## See recognize_whisper, possible to stream recognition
 #https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py


 # consider detecting language and having a global preference shared with the
 # speech generator, so that the system can infer language and speak back in the native
 # tongue of each request

    def InferFromFile(self, file):
        return self.pipe(
            file,
            chunk_length_s=30,
            batch_size=24,
            #return_timestamps="word", # set to True to get real timestamps for training sets
        )

    def Infer(self, audio_data):

        assert isinstance(audio_data, AudioData), "Data must be AudioData from SpeechRecognition"

        # 16 kHz
        wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream, dtype='float32')

        return self.pipe(
            audio_array,
            #task="translate"
        )
