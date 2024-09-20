from os import system
import sys
import warnings
import time
#from alsaerrorfilter import noalsaerr

from threading import Event

import concurrent.futures

keepListening = True
language="en"
speaker = 9

speechCancelToken=None


# Mic and SR wrapper engine to manage the audio stream
#############################################
# defaults -> sample_rate: 16000, chunk_size: 1024
import speech_recognition as sr 

recognizer=sr.Recognizer()
source = sr.Microphone()

print('Initializing TTS model')
import sox_speech
tts_manager = sox_speech.InferenceManager(language=language)
print ('TTS model initialized')

print('Initializing STT model')
import sox_hearing
stt_manager = sox_hearing.InferenceManager()
print('STT model initialized')

print('Initializing LLM model')
import sox_language
llm_manager = sox_language.InferenceManager(language=language)
print ('LLM model initialized')


def do_prompt(recognizer, audio):
    global keepListening
    global speechCancelToken

    textRequest = stt_manager.Infer(audio)
    request = textRequest['text'].strip()

    # we only want to process if there is enough material for inference
    ## otherwise, it is probably a false positive 
    ## 'thank you' or 'Okay' from the model
    if keepListening == True and len(request.split(' ')) > 2 or '?' in request:
        prompted = True
        print("User: {0}".format(request))
    else:
        print("Noise: {0}".format(request))

    if prompted:
        if speechCancelToken is not None:
            speechCancelToken.set()

        if keepListening:
            result = llm_manager.Infer(request)
            speechCancelToken=Event()

            tts_manager.Infer(result, speechCancelToken, speaker=speaker)


def start_listening():
    global keepListening

    with source as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=1)

    tts_manager.Infer('call me socks', speaker=speaker)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while keepListening:
            with source as mic:
                audio = recognizer.listen(mic)
                executor.submit(do_prompt, recognizer, audio)


if __name__ == '__main__':
    start_listening()