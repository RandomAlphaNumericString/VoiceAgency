import time
import speech_recognition as sr 
recognizer=sr.Recognizer()
source = sr.Microphone()

def write_wav(thing, audio):
    print("writing the wav file")
    with open("speech.wav", "wb") as f:
        f.write(audio.get_wav_data())

print("Normalizing for ambient noise")
with source as s:
    recognizer.adjust_for_ambient_noise(s, duration=2)

stopListening = recognizer.listen_in_background(source, write_wav)

print("Acquiring for 10 seconds")
time.sleep(10)

print("stopping listening and shutting down")
stopListening(wait_for_stop=False)