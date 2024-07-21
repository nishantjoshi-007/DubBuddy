from .youtube_download import YotubeDownloader
from .audio_process import AudioProcess
from .translation_process import TranslationProcess
from .lang_codes import language_code_mapping
from .text_to_speech import TextToSpeech
from .subtitles import Subtitle
from .video_process import VideoProcess
from ..util.util import cleanup
import asyncio, logging

#configured logging file
logging.basicConfig(filename='logs/combine.log', level=logging.INFO)

async def main(url, base_path, from_lang, to_lang, tos_check, save_translated_video_path):
    url = str(url)
    
    #from lang code 
    from_language = from_lang
    from_lang_code = language_code_mapping.get(from_language)
    print(from_lang_code)

    #to lang code 
    to_language = to_lang
    to_lang_code = language_code_mapping.get(to_language)
    print(to_lang_code)
    
    try:
        video_downloader = YotubeDownloader(url, base_path)
        
        #download video
        original_video_file = video_downloader.video_download()
        
        #download audio
        original_audio_file = video_downloader.audio_download()
        
        #get title
        video_title = video_downloader.title_extract()
        
        #get unique dir
        unique_directory = video_downloader.get_unique_directory()

        #audio to text
        text_converter = AudioProcess(original_audio_file, unique_directory, video_title)
        original_text_file = text_converter.audio_to_text()

        #translate text
        if original_text_file and from_lang_code and to_lang_code is not None:                
                #translate text
                text_translator = TranslationProcess(unique_directory, original_text_file, from_lang_code, to_lang_code, video_title)
                translated_text_file = text_translator.translate()
    
                if translated_text_file:
                    
                    #text to soeech
                    audio_converter = TextToSpeech(translated_text_file, original_audio_file, to_lang_code, unique_directory, video_title, tos_check)
                    translated_audio_file = audio_converter.text_to_audio()

                    #translated subtitle
                    if translated_audio_file:
                        
                        #subtitle generator    
                        subtitle_generator = Subtitle(translated_audio_file, unique_directory, video_title, to_lang_code)
                        translated_subtitle_file = subtitle_generator.generate_subtitles()

                        if translated_subtitle_file:

                            #merge video
                            video_merger = VideoProcess(original_video_file, translated_audio_file, translated_subtitle_file, unique_directory, video_title)
                            translated_video_file = video_merger.video_merger()
                            
                            # Store the final video path in the callback
                            save_translated_video_path(translated_video_file)
        
        #sleep for certain time
        await asyncio.sleep(300)
        
        # delete unique directory       
        cleanup(unique_directory)
        logging.info("procces has been completed and unique directory has been deleted.")

    except Exception as e:
        logging.error(f"Error in final method: {e}")
        raise Exception(f"Error in final method.")