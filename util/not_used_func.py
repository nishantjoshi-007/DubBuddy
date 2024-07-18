#pytube method to download video 
#from pytube import YouTube

#audioclip method to extract audio from video
from moviepy.editor import VideoFileClip
import os

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
        

#video download
def video_download(video_url, original_video_path, pytube):
    try:
        #check for the url
        if video_url is None:
            print("Video URL can not be empty.")
            return None
        
        #get the video and download
        video = pytube.YouTube(video_url)
        video_quality = video.streams.get_highest_resolution()
        video_quality.download(output_path = original_video_path)
        print(f'Downloaded Successfully {video_url}')

    except Exception as e:
        print(f'Error Downloading the video {video_url}. Reason: {e}')
        

#function to extract audio from video:
def video_to_audio(original_video_file_path, original_audio_path, video_title):
    try:
        #get audio from video
        print("Audio seperation started...")
        video = VideoFileClip(original_video_file_path)
        audio = video.audio
        audio_filename = os.path.join(original_audio_path, f"{video_title}.wav")
        if audio:
            audio.write_audiofile(audio_filename)
            print(f"Audio seperation completed. Audio saved to {original_audio_path}")
            return audio_filename
    
    except OSError as e:
        print(f"Error: The video file could not be found or opened. {e}")
        return None
    
    except Exception as e:
        print(f"Error: There was an issue writing the audio file to disk. {e}")
        return None

#websockets example
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     tasks = [
#         ("Downloading video", combine.YotubeDownloader),
#         ("Extracting audio", combine.YotubeDownloader),
#         ("Converting audio to text", combine.AudioProcess),
#         ("Translating text", combine.TranslationProcess),
#         ("Converting text to audio", 3),
#         ("Merging audio and video", 2)
#     ]
    
#     for index, (task, duration) in enumerate(tasks):
#         await websocket.send_json({"step": index + 1, "total_steps": len(tasks), "message": task})
#         time.sleep(duration)
    
#     await websocket.send_json({"step": len(tasks), "total_steps": len(tasks), "message": "Completed"})