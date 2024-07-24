import yt_dlp
import os, uuid
import logging
from ..util.util import sanitize_filename

class YotubeDownloader:

    #initialize and define paths
    def __init__(self, url:str, base_dir_path:str) -> None:
        self.url = url
        self.title = None

        self.base_dir_path = base_dir_path
        self.unique_dir_path = os.path.join(self.base_dir_path, self.unique_id())
        
        self.video_dir_path = os.path.join(self.unique_dir_path, "original_video")
        self.audio_dir_path = os.path.join(self.unique_dir_path, "original_audio")

        # Create the directories for video and audio
        os.makedirs(self.unique_dir_path, exist_ok=True)
        os.makedirs(self.video_dir_path, exist_ok=True)
        os.makedirs(self.audio_dir_path, exist_ok=True)




    #method to download video    
    def video_download(self) -> str:
        try:
            ydl_opts = {
                "format" : "bestvideo/best",
                "outtmpl": f"{self.video_dir_path}/%(title)s.%(ext)s",
                "retries": 5,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.url, download=True)
                video_file = ydl.prepare_filename(info_dict)
                return video_file

        except Exception as e:
            logging.error(f"Error during downloading video from youtube: {e}")
            raise Exception("Error during downloading video from youtube.")




    #method to download audio
    def audio_download(self) -> str:
        try:
            ydl_opts = {
                "format" : "bestaudio/best",
                "outtmpl": f"{self.audio_dir_path}/%(title)s.%(ext)s",
                "retries": 5,
            }
        
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.url, download=True)
                audio_file = ydl.prepare_filename(info_dict)
                return audio_file

        except Exception as e:
            logging.error(f"Error during downloading audio from youtube: {e}")
            raise Exception("Error during downloading audio from youtube.")




    #method to extract title and return it            
    def title_extract(self) -> str:
        try:
            ydl_opts = {
                "format" : "best",
                "noplaylist" : True,
                "quiet" : True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.url, download=False)
        
                if info_dict is not None:
                    youtube_title = info_dict.get("title", "No title found.")
                    self.title = sanitize_filename(youtube_title)
                else:
                    self.title = "No title found."
                
                return self.title

        except Exception as e:
            logging.error(f"Error during downloading title from youtube: {e}")
            raise Exception("Error during downloading title from youtube.")


        
    #returns the unique dir
    def get_unique_directory(self) -> str:
        try:
            return self.unique_dir_path
        except Exception as e:
            logging.error(f"Error during getting unique directory: {e}")
            raise Exception("Error during getting unique directory.")
   

    #generate unique id for each process
    def unique_id(self) -> str:
        try:
            return str(uuid.uuid4())
        except Exception as e:
            logging.error(f"Error during generating unique id: {e}")
            raise Exception("Error during generating unique id.")