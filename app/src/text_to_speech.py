from TTS.api import TTS
import torch
import os, logging
from ..util.util import convert_audio_to_wav

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
            logging.error("Terms of Service not agreed to.")
            raise ValueError("Terms of Service not agreed to.")
        
        #read the file        
        try:
            with open(self.translated_text_file , "r" , encoding = "utf-8") as f:
                translated_text = f.read()
        except FileNotFoundError:
            logging.error(f"Error: The file {self.translated_text_file} was not found.")
            raise FileNotFoundError(f"The file {self.translated_text_file} was not found.")


        #convert audio in wav format        
        wav_audio_file = convert_audio_to_wav(self.audio_file, self.unique_dir_path, self.title)
        if not wav_audio_file:
            logging.error("Error converting audio to wav format.")
            raise Exception("Error converting audio to wav format.")


        #get the model, start text to speech with original voice and translated text
        try:
            text_to_speech_audio_file = os.path.join(self.translated_audio_dir_path, f"{self.title}.wav")
            translated_audio_file = os.path.join(self.translated_audio_dir_path, f"{self.title}finalaudio.wav")
            
            # Get device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logging.info("Model loaded and text to speech started successfully.")
            
            # Text to speech
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False).to(device)
            tts.tts_to_file(
                text = translated_text,
                file_path = text_to_speech_audio_file,
                speaker_wav = wav_audio_file,
                language = self.to_lang_code
            )
            
            # Voice conversion
            tts_vc = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)
            tts_vc.voice_conversion_to_file(
                source_wav = translated_audio_file, 
                target_wav = wav_audio_file, 
                file_path = translated_audio_file
            )
            
            return translated_audio_file
            
        except Exception as e:
            logging.error(f"Error during Text to Speech conversion: {e}")
            raise Exception(f"Error during Text to Speech conversion.")