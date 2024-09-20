#VoiceAgency
#License [Apache](./LICENSE.txt)
VoiceAgency is a simple project that plumbs together speech recognition, LLMs, and text-to-speech to create simple voice agents.  It is a fun way to explore human-computer interaction as it evolves with machine-learning techniques.

The goal of the project is to create a simple baseline that can be used to explore applications for speech technology.  We want to improve the individual components and make it easy for people to build voice agents.

#Setup
 * Install [Anaconda](https://www.anaconda.com/download)
 * Open "anaconda prompt" and create a new environment
    ```conda create -n soxenv python=3.11.5 pip setuptools wheel```
 * Install dependencies
    ```pip install -i requirements.txt```
 * Try it out
    ```python main.py```

#Exploration
In addition to main.py, there are modules to test acquisition, speech, and hearing.  These can be used to locally test those components or as an example of how to use those components directly.
* test_acquisition.py attempts to use SpeechRecognition to detect voice activity and captures a wav sample (speech.wav)
* test_hearing.py opens a file called speech.wav and transcribes it
* test_speech.py emits speech for a fixed phrase to verify speech output
