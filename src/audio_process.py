import whisper 
import os

class AudioProcess:

    #initialize and define paths        
    def __init__(self, audio_file:str, unique_dir_path:str, title:str) -> None:
        self.audio_file = audio_file
        self.title = title
        self.unique_dir_path = unique_dir_path
        
        self.text_dir_path = os.path.join(self.unique_dir_path, "original_text")
        os.makedirs(self.text_dir_path, exist_ok=True) # Create the directory for text


    #load whisper model
    def get_model(self):
        try:
            print("loading the whisper model")
            model = whisper.load_model("base")
            return model
        except Exception as e:
            print(f"Error during downloading the model: {e}")
        
        
    #function to convert the audio to text:
    def audio_to_text(self):
        try:            
            #load the model, if downloaded, it will be stored in cache
            model = self.get_model()
            print("model loaded successfully.")
            
            #start transcribing - text from audio
            if model:
                print(f"this is the audio path: {self.audio_file}")
                result = model.transcribe(self.audio_file, fp16=False)
                
                #create a txt file from audio
                print("Transcription started...")
                text_file = os.path.join(self.text_dir_path, f"{self.title}.txt")
                with open(text_file, "w") as f:
                    f.write(str(result["text"]))
                print(f"Transcription completed. Text saved to {text_file}")
                return text_file

        except Exception as e:
            print(f"Error during transcription or file writing: {e}")
            return None