#pip install soundfile ffmpeg
#https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-6.1.1-essentials_build.zip

import sox_hearing

print("initializing hearing")
im = sox_hearing.InferenceManager()

print("inferring from speech.wav")
result = im.InferFromFile("speech.wav")
print("inferred...")
print(result["text"])
