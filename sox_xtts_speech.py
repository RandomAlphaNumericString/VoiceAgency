import warnings
import time

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import simpleaudio as sa

if torch.cuda.is_available():
    torch_dtype = torch.float16 
    worker_device = "cuda:0"
else:
    torch_dtype = torch.float32
    worker_device="cpu"


#######################################################################################
# Name: TTSInferenceProcessManager
#
# Purpose:  Encapsulates loading/managing TTS model state as well as executing inference.
# 
# Caveats:  Currently single-language.  This will need a factory with specializations
#  per-language and a base-class/generic method for Infer in the long-term or we will
#  need to pass language through to a multi-lingual TTST (text to speech translation) model
#
#######################################################################################
class InferenceManager:

    def __init__(self, language, template):
        self.tts_model = self.getModelForLanguage(language)
        self.template = template
        self.language = language


    def getModelForLanguage(self, language):
        #TTS Coqui (XTTS-v2)
        self.tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        return TTS(self.tts_model_name).to(worker_device)


    def Infer(self, textRequest, cancelToken=None): 
            print(textRequest)
            self.tts_model.tts_to_file(
                text=textRequest,
                speaker_wav=self.template,
                language=self.language,
                file_path="speechOutput.wav",
            )
            print("speaking ")
            wave_obj = sa.WaveObject.from_wave_file("speechOutput.wav")
            play_obj = wave_obj.play()

            while play_obj.is_playing():
                if cancelToken is not None and cancelToken == True:
                    play_obj.stop()
                time.sleep(.2) # 200 milliseconds
            print("done speaking")
