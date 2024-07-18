from moviepy.editor import VideoFileClip , ImageClip , AudioFileClip , CompositeVideoClip
from moviepy.video.fx.speedx import speedx
import cv2
import numpy as np    
import srt
from datetime import timedelta
import os

class VideoProcess:

    #initialize and define paths
    def __init__(self, video_file:str, translated_audio_file:str, translated_subtitle_file:str, unique_dir_path:str, title:str) -> None:
        self.video_file = video_file
        self.translated_audio_file = translated_audio_file
        self.translated_subtitle_file = translated_subtitle_file

        self.unique_dir_path = unique_dir_path
        self.title = title

        self.translated_video_dir_path = os.path.join(self.unique_dir_path, "translated_video")
        os.makedirs(self.translated_video_dir_path, exist_ok=True) # Create the directory for translated audio


    def video_merger(self) -> str | None:
        #loading the video and audio
        video = VideoFileClip(self.video_file)
        translated_audio = AudioFileClip(self.translated_audio_file)

        #calculate the speed factor
        speed_factor = translated_audio.duration / video.duration
        print("speed factor is: ", speed_factor)

        #sync the audio and subtitles if they are not same as the video length
        if speed_factor != 1:
            #adjusting audio
            translated_audio = speedx(translated_audio, factor = speed_factor)
            
            #adjusting subtitles
            with open(self.translated_subtitle_file, 'r') as file:
                subtitles = list(srt.parse(file.read()))

            with open(self.translated_subtitle_file, 'w') as file:
                for sub in subtitles:
                    #scale the start and end times
                    sub.start = timedelta(seconds=sub.start.total_seconds() / speed_factor)
                    sub.end = timedelta(seconds=sub.end.total_seconds() / speed_factor)
                    file.write(srt.compose([sub]))

        
        #loading subtitles
        with open(self.translated_subtitle_file , 'r') as f:
            subtitle_generator = srt.parse(f.read())
            subtitles = list(subtitle_generator)
            subtitle_clips = []
            
        for sub in subtitles:
            img = np.zeros((100, video.size[0], 3), dtype=np.uint8)
            cv2.putText(img, sub.content, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            clip = ImageClip(img).set_duration(sub.end.total_seconds() - sub.start.total_seconds()).set_start(sub.start.total_seconds())
            subtitle_clips.append(clip)

            #Adding subtitles in the video
            video_with_subtitle = CompositeVideoClip([video] + subtitle_clips)
                
            #Replacing the original audio in the video with translated audio
            video_with_audio = video_with_subtitle.set_audio(translated_audio)
                
            #Final output video
            final_video_file = os.path.join(self.translated_video_dir_path, f"{self.title}.mp4")
            
            video_with_audio.write_videofile(final_video_file, codec="libx264")
            return final_video_file