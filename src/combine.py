from .youtube_download import YotubeDownloader
from .audio_process import AudioProcess
from .translation_process import TranslationProcess
from .lang_codes import language_code_mapping
from .text_to_speech import TextToSpeech
import time


def final_method(url, base_path, from_lang, to_lang, tos_check):
    url = str(url)
    
    try:
        video = YotubeDownloader(url, base_path)
        
        #download video
        original_video_file = video.video_download()
        print(f"video is saved here: {original_video_file}")
        
        #download audio
        original_audio_file = video.audio_download()
        print(f"audio is saved here: {original_audio_file}")    
        
        #get title
        video_title = video.title_extract()
        print(f"title is this: {video_title}")
        
        #get unique dir
        unique_directory = video.get_unique_directory()
        print(f"unique directory path is: {unique_directory}")
        
        #audio to text
        audio = AudioProcess(original_audio_file, unique_directory, video_title)
        
        original_text_file = audio.audio_to_text()
        print(f"text file is saved here: {original_text_file}")
        
        #translate text
        if original_text_file:
            
            #from lang code 
            from_language = from_lang
            from_lang_code = language_code_mapping.get(from_language)

            #to lang code 
            to_language = to_lang
            to_lang_code = language_code_mapping.get(to_language)
            
            if from_lang_code and to_lang_code:
                translator = TranslationProcess(unique_directory, original_text_file, from_lang_code, to_lang_code, video_title)
                translated_text_file = translator.translate()
                print(f"translated text file is here: {translated_text_file}")   
                
                if translated_text_file:
                    audio_translator = TextToSpeech(translated_text_file, original_audio_file, to_lang_code, unique_directory, video_title, tos_check)
                    translated_audio_file = audio_translator.text_to_audio()
                    print(f"translated audio is here: {translated_audio_file}")






        
        #sleep
        print("sleeping for 30 seconds.")
        time.sleep(30)
        
        #delete directory       
        #video.cleanup()
        print("directory has been deleted.")

    except Exception as e:
        print(f"Error in final method: {e}")