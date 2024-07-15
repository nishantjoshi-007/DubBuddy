import torch
import torchaudio
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
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

            if wav_audio_file:
                print("text to speech started.")

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')

                tts_model = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch", map_location=device)
                vocoder_model = HifiGanModel.from_pretrained(model_name="tts_hifigan", map_location=device)

                speaker_wav = wav_audio_file
                speaker_embedding = tts_model.extract_speaker_embeddings(speaker_wav)

                parsed = tts_model.parse(translated_text)
                spectrogram = tts_model.generate_spectrogram(tokens=parsed, speaker_embeddings=speaker_embedding)
                speech = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)

                torchaudio.save(translated_audio_file, speech, sample_rate=22050)
                print(f"Text-to-speech conversion completed. Audio saved to {translated_audio_file}")
                return translated_audio_file

        except Exception as e:
            print(f"Error during Text to Speech conversion: {e}")