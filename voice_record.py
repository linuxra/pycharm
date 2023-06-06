import speech_recognition as sr
from pydub import AudioSegment

# create a recognizer
r = sr.Recognizer()

# define the microphone
with sr.Microphone() as source:
    print("Please say something...")
    # listen for the first phrase and extract it into audio data
    audio_data = r.record(source, duration=5) # you can change duration to the desired value
    print("Recognizing...")

    # convert speech to text
    text = r.recognize_google(audio_data)
    print(text)

# writing audio data into .wav file
with open("output.wav", "wb") as f:
    f.write(audio_data.get_wav_data())
