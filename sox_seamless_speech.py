import warnings
import time
import sys
import warnings
import nltk
import nltk.data


if not sys.warnoptions:
    warnings.simplefilter("ignore")

from transformers import AutoProcessor, SeamlessM4Tv2ForTextToSpeech
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
        nltk.download('punkt')
        self.language = self.mapLanguage(language)
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2ForTextToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
        self.sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')


    def mapLanguage(self, language):
        if 'en' in language.lower():
            return 'eng'
        else:
            return ''


    def Infer(self, textRequest, cancelToken=None, speaker=0): 

        sampleRate = self.model.config.sampling_rate

        print(textRequest)
        sentences = self.sentence_detector.tokenize(textRequest.strip())

        play_obj = None

        for sentence in sentences:
            # try to detect cancelation before inference
            if cancelToken is not None and cancelToken.is_set():
                if play_obj is not None:
                    play_obj.stop()
                break

            text_inputs = self.processor(text = sentence, src_lang="eng", return_tensors="pt")
            audio_array = self.model.generate(**text_inputs, speaker_id=speaker, tgt_lang=self.language)[0].cpu().numpy().squeeze()

            #we may run up to one inference ahead while playback runs behind
            # todo, possibly increase this buffer count if needed for smoothing
            #        currently the buffered item is the play_obj, but we would need
            #  to push the abstraction down to the audio_array and model sa as an executor
            # with play_obj as the current executing buffer and the rest of the queue being
            # represented as the wav buffers (audio_array instances returned by generate)
            shouldAbort = False
            while play_obj is not None and play_obj.is_playing():
                if cancelToken is not None and cancelToken.is_set():
                    play_obj.stop()
                    shouldAbort = True
                    break
                time.sleep(.2) # 200 milliseconds

            if shouldAbort:
                break

            play_obj = sa.play_buffer(audio_array, 1, 4, sampleRate)
