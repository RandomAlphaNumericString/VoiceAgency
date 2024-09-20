#VoiceAgency
#License Apache [./LICENSE.txt]
VoiceAgency is a simple project that plumbs together speech recognition, LLMs, and text-to-speech to create simple voice agents.  It is a fun way to explore human-computer interaction as it evolves with machine-learning techniques.

The goal of the project is to create a simple baseline that can be used to explore applications for speech technology.


#Setup
 * Install Anaconda [https://www.anaconda.com/download]
 * Open "anaconda prompt" and create a new environment
    conda create -n soxenv python=3.11.5 pip setuptools wheel
 * Install dependencies
    pip install -i requirements.txt
 * Try it out
    python main.py

#Exploration
In addition to main.py, there are modules to test acquisition, speech, and hearing.  These can be used to locally test those components or as an example of how to use those components directly.