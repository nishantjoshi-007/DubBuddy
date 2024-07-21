import whisper 
import os, logging

class AudioProcess:

    #configured logging file
    logging.basicConfig(filename='logs/audio_process.log', level=logging.INFO)

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
            model = whisper.load_model("large")
            return model
        except Exception as e:
            logging.error(f"Error during loading the model: {e}")
            raise Exception("Error during loading the model.")
        
        
    #function to convert the audio to text:
    def audio_to_text(self):
        try:            
            #load the model, if downloaded, it will be stored in cache
            model = self.get_model()
            
            #start transcribing - text from audio
            if model:
                result = model.transcribe(self.audio_file, fp16=False)
                
                #create a txt file from audio
                text_file = os.path.join(self.text_dir_path, f"{self.title}.txt")
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(str(result["text"]))
                return text_file

        except Exception as e:
            logging.error(f"Error during converting the audio to text: {e}")
            raise Exception("Error during converting the audio to text.")