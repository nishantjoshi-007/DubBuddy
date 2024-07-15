from pydub import AudioSegment
import os, shutil, logging

#configured logging file
logging.basicConfig(filename='logs/utils.log', level=logging.INFO)

# Convert audio to WAV format
def convert_audio_to_wav(audio_path:str, unique_dir_path:str, title:str) -> str | None:
    try:
        audio = AudioSegment.from_file(audio_path)
        wav_path = os.path.join(unique_dir_path, "original_audio", f"{title}.wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        logging.error(f"Error during audio conversion: {e}")
        raise Exception(f"Error during audio conversion.")    
    
#cleanup once process is done
def cleanup(unique_dir_path:str) -> None:
    try:
        shutil.rmtree(unique_dir_path)
    except Exception as e:
        logging.error(f"Error during deleting the unique folder: {e}")
        raise Exception(f"Error during deleting the unique folder.")        
         
#function to format timestamps
def format_timestamp(time_in_seconds):
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

#remove invalid characters from filename
def sanitize_filename(filename):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', "'"]
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename

#check if directory exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        

#pytube method to download video 
from pytube import YouTube

#video download
def video_download(video_url, original_video_path):
    try:
        #check for the url
        if video_url is None:
            print("Video URL can not be empty.")
            return None
        
        #get the video and download
        video = YouTube(video_url)
        video_quality = video.streams.get_highest_resolution()
        video_quality.download(output_path = original_video_path)
        print(f'Downloaded Successfully {video_url}')

    except Exception as e:
        print(f'Error Downloading the video {video_url}. Reason: {e}')
        

# video to audio conversion using moviepy  
from moviepy.editor import VideoFileClip
import os

#function to extract audio from video:
def video_to_audio(original_video_file_path, original_audio_path, video_title):
    try:
        #get audio from video
        print("Audio seperation started...")
        video = VideoFileClip(original_video_file_path)
        audio = video.audio
        audio_filename = os.path.join(original_audio_path, f"{video_title}.wav")
        audio.write_audiofile(audio_filename)
        print(f"Audio seperation completed. Audio saved to {original_audio_path}")
        return audio_filename
    
    except OSError as e:
        print(f"Error: The video file could not be found or opened. {e}")
        return None
    
    except Exception as e:
        print(f"Error: There was an issue writing the audio file to disk. {e}")
        return None