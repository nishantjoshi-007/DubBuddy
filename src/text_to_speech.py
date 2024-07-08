from TTS.api import TTS
import wexpect
import os
from ..util.util import convert_audio_to_wav

#encoding - import sys
#sys.stdout.reconfigure(encoding="utf-8")

class TextToSpeech:

    #initialize and define paths
    def __init__(self, translated_text_file:str, audio_file:str, to_lang_code:str, unique_dir_path:str, title:str, tos_check:bool) -> None:
        self.translated_text_file = translated_text_file
        self.audio_file = audio_file
        
        self.to_lang_code = to_lang_code
        self.tos_check = tos_check
        self.title = title
         
        self.unique_dir_path = unique_dir_path

        self.translated_audio_dir_path = os.path.join(self.unique_dir_path, "translated_audio")
        os.makedirs(self.translated_audio_dir_path, exist_ok=True) # Create the directory for translated audio



    
    #text to speech function
    def text_to_audio(self) -> str | None:
        #check for terms of service
        if not self.tos_check:
            print("Terms of Service not agreed to.")
            return None
        
        #read the file        
        try:
            with open(self.translated_text_file , "r" , encoding = "utf-8") as f:
                translated_text = f.read()
        except FileNotFoundError:
            print(f"Error: The file {self.translated_text_file} was not found.")
            return


        #convert audio in wav format        
        wav_audio_file = convert_audio_to_wav(self.audio_file, self.unique_dir_path, self.title)
        if not wav_audio_file:
            print("Error converting audio to wav format.")
            return


        #get the model, start text to speech with original voice and translated text
        try:
            translated_audio_file = os.path.join(self.translated_audio_dir_path, f"{self.title}.wav")

            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            print("model loaded successfully.")
            
            #to download the model
            command = "python -c 'from TTS.api import TTS; tts = TTS(\"tts_models/multilingual/multi-dataset/xtts_v2\", gpu=False)'"
            child = wexpect.spawn(command, encoding='utf-8', timeout=120)
            child.expect("Otherwise, I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml")
            child.sendline("y")
            
            if wav_audio_file:
                print("text to speech started.")
                tts.tts_to_file(
                    text = translated_text,
                    file_path = translated_audio_file,
                    speaker_wav = wav_audio_file,
                    language = self.to_lang_code
                )
        
        except Exception as e:
            print(f"Error during Text to Speech conversion: {e}")