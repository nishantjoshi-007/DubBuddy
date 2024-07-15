from argostranslatefiles import argostranslatefiles
import argostranslate.package
import argostranslate.translate
import argostranslatefiles
import os, logging

class TranslationProcess:

    #configured logging file
    logging.basicConfig(filename='logs/translation_process.log', level=logging.INFO)
    
    #initialize and define paths
    def __init__(self, unique_dir_path:str, text_file:str, from_lang:str, to_lang:str, title:str) -> None:
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.title = title
        
        self.text_file = text_file
        self.unique_dir_path = unique_dir_path

        self.translated_text_dir_path = os.path.join(self.unique_dir_path, "translated_text")
        os.makedirs(self.translated_text_dir_path, exist_ok=True) # Create the directory for translated text

    # Download translation package
    def translation_package(self):
        # Getting the list of installed languages and checking for matching language code
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang_installed = any(lang.code == self.from_lang for lang in installed_languages)
        to_lang_installed = any(lang.code == self.to_lang for lang in installed_languages)
        
        if from_lang_installed and to_lang_installed:
            # Check if the translation package for from_language to to_language exists
            from_lang = next((lang for lang in installed_languages if lang.code == self.from_lang), None)
            to_lang = next((lang for lang in installed_languages if lang.code == self.to_lang), None)
            
            # Check if the translation package is already installed
            if from_lang and to_lang and from_lang.get_translation(to_lang):
                logging.info(f"Translation package from {self.from_lang} to {self.to_lang} is already installed.")
                return from_lang.get_translation(to_lang)
        
        # If the package exists but is not installed
        logging.info(f"Installing translation package from {self.from_lang} to {self.to_lang}...")
        
        # Update the package index and get the package
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == self.from_lang and x.to_code == self.to_lang, available_packages
            ), None
        )
        
        # Install the package
        if package_to_install is not None:
            package_path = package_to_install.download()
            argostranslate.package.install_from_path(package_path)
            logging.info(f"Installed translation package from {self.from_lang} to {self.to_lang}.")        
        else:
            logging.error(f"No available translation package from {self.from_lang} to {self.to_lang}.")
            raise Exception(f"No available translation package from {self.from_lang} to {self.to_lang}.")

        # Retrieve the language pair again after installation
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = list(filter(
            lambda x: x.code == self.from_lang, installed_languages))[0]
        to_lang = list(filter(
            lambda x: x.code == self.to_lang, installed_languages))[0]
        underlying_translation = from_lang.get_translation(to_lang)
        return underlying_translation




    # Translation function to translate
    def translate(self) -> str | None:
        try:
            # Read input file
            if self.text_file:
                with open(self.text_file, "r", encoding="utf-8") as file:
                    original_text = file.read()
                logging.info("Translation started...")

                # Translate text
                underlying_translation = self.translation_package()
            
                if underlying_translation:
                    translated_text = underlying_translation.translate(original_text)
            
                # Write translated output file
                translated_text_file = os.path.join(self.translated_text_dir_path, f"{self.title}.txt")
                with open(translated_text_file, "w", encoding="utf-8") as f:
                    f.write(translated_text)
                logging.info("Translation completed.")
                return translated_text_file
        
        except Exception as e:
            logging.error(f"Unable to translate: {e}")
            raise Exception("Unable to translate.")