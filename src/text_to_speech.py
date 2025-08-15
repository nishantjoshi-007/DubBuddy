from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
import os
from ..util.util import convert_audio_to_wav
from dotenv import load_dotenv

class TextToSpeech:
    
    load_dotenv()  # Load environment variables from .env file

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

        # Initialize Fish Audio session with your API key
        api_key = os.getenv("FISH_API_KEY")
        if not api_key:
            raise ValueError("API key for Fish Audio is not set in the environment variables.") 
        self.session = Session(api_key)  # Replace with your actual API key

    
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

        try:
            translated_audio_file = os.path.join(self.translated_audio_dir_path, f"{self.title}finalaudio.wav")

            # Read the reference audio file
            with open(wav_audio_file, "rb") as audio_file:
                audio_data = audio_file.read()

            # Use Fish Audio TTS with voice cloning
            # This combines both text-to-speech and voice cloning in one call
            with open(translated_audio_file, "wb") as output_file:
                for chunk in self.session.tts(TTSRequest(
                    text=translated_text,
                    references=[
                        ReferenceAudio(
                            audio=audio_data,
                            text=""  # Leave empty for automatic transcription
                        )
                    ]
                ),
                    backend="speech-1.6"):
                    output_file.write(chunk)

            print(f"Fish Audio TTS with voice cloning completed. Audio saved to {translated_audio_file}")
            return translated_audio_file

        except Exception as e:
            print(f"Error during Fish Audio TTS conversion: {e}")
            return None