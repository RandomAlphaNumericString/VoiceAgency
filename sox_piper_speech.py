import warnings
import time
import sys
import warnings

from piper import PiperVoice


if not sys.warnoptions:
    warnings.simplefilter("ignore")

import simpleaudio as sa

#######################################################################################
# Name: TTSInferenceProcessManager
#
# Purpose:  Encapsulates loading/managing TTS model state as well as executing inference.
# 
# Caveats:  Currently using M4T-V2, which is a single model translation library taking
#  target language as a parameter.  If there is a need for multiple distinct models
#  on a per-language basis, we will need a factory and specializations.
#
#######################################################################################
class InferenceManager:

    def __init__(self, language):
        self.language = self.mapLanguage(language)
        self.model = self.getModelForLanguage(language)

    def mapLanguage(self, language):
        if 'en' in language.lower():
            return 'eng'
        else:
            return ''

    def getModelForLanguage(self, language):
        if 'en' in language.lower():
            model_path  = "en_GB-alba-medium.onnx"
            config_path = "en_GB-alba-medium.onnx.json"
            return PiperVoice.load(model_path, config_path)
        else:
            raise Exception("unsupported language in speech module: ", language)


    def Infer(self, textRequest, cancelToken=None, speaker=0): 

        print(textRequest)

        shouldContinue = True
        for audio_array in self.model.synthesize_stream_raw( text=textRequest):

            # mono, 16 bit per:
            # https://github.com/rhasspy/piper/blob/master/src/python_run/piper/voice.py
            play_obj = sa.play_buffer(audio_array, 1, 2, self.model.config.sample_rate)
            while play_obj.is_playing():
                if cancelToken is not None and cancelToken.is_set():
                    play_obj.stop()
                    shouldContinue = False
                    break
                time.sleep(.2) # 200 milliseconds

            if shouldContinue == False:
                break;

