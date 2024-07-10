from .youtube_download import YotubeDownloader
from .audio_process import AudioProcess
from .translation_process import TranslationProcess
from .lang_codes import language_code_mapping
from .text_to_speech import TextToSpeech
from .subtitles import Subtitle
from .video_process import VideoProcess
from ..util.util import cleanup
import time


def final_method(url, base_path, from_lang, to_lang, tos_check):
    url = str(url)
    
    try:
    #################################################################################################### 
        video_downloader = YotubeDownloader(url, base_path)

        #download video
        original_video_file = video_downloader.video_download()
        print(f"video is saved here: {original_video_file}")
        
        #download audio
        original_audio_file = video_downloader.audio_download()
        print(f"audio is saved here: {original_audio_file}")    
        
        #get title
        video_title = video_downloader.title_extract()
        print(f"title is this: {video_title}")
        
        #get unique dir
        unique_directory = video_downloader.get_unique_directory()
        print(f"unique directory path is: {unique_directory}")

    ####################################################################################################         
        #audio to text
        audio_converter = AudioProcess(original_audio_file, unique_directory, video_title)
        original_text_file = audio_converter.audio_to_text()
        print(f"text file is saved here: {original_text_file}")
        
        #translate text
        if original_text_file:
            
            #from lang code 
            from_language = from_lang
            from_lang_code = language_code_mapping.get(from_language)

            #to lang code 
            to_language = to_lang
            to_lang_code = language_code_mapping.get(to_language)

    ####################################################################################################    
            if from_lang_code and to_lang_code:
                text_translator = TranslationProcess(unique_directory, original_text_file, from_lang_code, to_lang_code, video_title)
                translated_text_file = text_translator.translate()
                print(f"translated text file is here: {translated_text_file}")   
    
    ####################################################################################################             
                if translated_text_file:
                    audio_translator = TextToSpeech(translated_text_file, original_audio_file, to_lang_code, unique_directory, video_title, tos_check)
                    translated_audio_file = audio_translator.text_to_audio()
                    print(f"translated audio is here: {translated_audio_file}")

    #################################################################################################### 
                    #translated subtitle
                    if translated_audio_file:
                        subtitle_generator = Subtitle(translated_audio_file, unique_directory, video_title, to_lang_code)
                        translated_subtitle_file = subtitle_generator.generate_subtitles()
                        print(f"translated subtitles are here: {translated_subtitle_file}")

    #################################################################################################### 
                        #merge video
                        if translated_subtitle_file:
                            video_merger = VideoProcess(original_video_file, translated_audio_file, translated_subtitle_file, unique_directory)
                            translated_video_file = video_merger.video_merger()
                            print(f"translated video with subtitle is stored here: {translated_video_file}")

        
        #sleep
        print("sleeping for 60 seconds.")
        time.sleep(60)
        
        #delete directory       
        cleanup(unique_directory)
        print("directory has been deleted.")

    except Exception as e:
        print(f"Error in final method: {e}")