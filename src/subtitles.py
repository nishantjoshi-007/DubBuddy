import os
import whisper
import whisper_timestamped as whisperTP
from ..util.util import format_timestamp

class Subtitle:
    #initialize and define paths
    def __init__(self, audio_file:str, unique_dir_path:str, video_title:str, subtitle_lang:str) -> None:
        self.audio_file = audio_file
        self.subtitle_lang = subtitle_lang
        
        self.video_title = video_title
        
        self.unique_dir_path = unique_dir_path

        self.subtitle_dir_path = os.path.join(self.unique_dir_path, "subtitles")
        os.makedirs(self.subtitle_dir_path, exist_ok=True) # Create the directory for translated audio


    #generate subtitles
    def generate_subtitles(self) -> str | None:
        try:
            #get model
            model_TP = whisperTP.load_model("base")
            print("Original Subtitle Transcription started...")
            audio_model = whisperTP.load_audio(self.audio_file)
            result = whisperTP.transcribe(model_TP, audio_model, language=self.subtitle_lang)
            
            #start writing srt file
            subtitle_file = os.path.join(self.subtitle_dir_path, f"{self.video_title}.srt")
            with open(subtitle_file, "w") as f:
                for i, segment in enumerate(result["segments"]):
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    f.write(f"{i+1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(segment["text"] + "\n\n")
            print(f"Original Subtitle Transcription completed. Subtitles saved to {subtitle_file}")
            
            return subtitle_file
        
        except Exception as e:
            print(f"Error during transcription or file writing: {e}")