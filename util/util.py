from pydub import AudioSegment
import os, shutil

# Convert audio to WAV format
def convert_audio_to_wav(audio_path:str, unique_dir_path:str, title:str) -> str | None:
    try:
        audio = AudioSegment.from_file(audio_path)
        wav_path = os.path.join(unique_dir_path, "original_audio", f"{title}.wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        return None
    
    
#cleanup once process is done
def cleanup(unique_dir_path:str) -> None:
    try:
        shutil.rmtree(unique_dir_path)
    except Exception as e:
        print(f"Error during deleting the unique folder: {e}")
        
        
        
#function to format timestamps
def format_timestamp(time_in_seconds):
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"