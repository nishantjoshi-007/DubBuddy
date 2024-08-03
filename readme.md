# :star2: DubBuddy
<div align='center'>
Welcome to the 'DubBuddy' GitHub Repository. DubBuddy is an innovative tool that simplifies the process of translating and dubbing videos into multiple languages using advanced voice cloning technology. This project extracts and processes video content, translating and dubbing it seamlessly while also generating subtitles. Perfect for content creators aiming to reach a global audience, DubBuddy offers a streamlined and efficient solution for multilingual video content. Explore the repository to dive into the code and see how DubBuddy can transform your videos.<h4><span> · </span> <a href="https://dub-buddy.com/"> Live Demo </a><span> · </span> <a href="https://github.com/nishantjoshi-007/DubBuddy/issues"> Report Bug </a> <span> · </span> <a href="https://github.com/nishantjoshi-007/DubBuddy/issues"> Request Feature </a></h4>
</div>

# :notebook_with_decorative_cover: Table of Contents
- [Getting Started](#toolbox-getting-started)
- [Features](#dart-features)
- [Contributing](#wave-contributing)
- [License](#warning-license)
- [Acknowledgements](#gem-acknowledgements)

## :toolbox: Getting Started

### :bangbang: Prerequisites
- Install Visual Studio Code as Code Editor<a href="https://code.visualstudio.com/Download"> Here</a>
- To set up this project, you'll need Python and pip installed on your machine.<a href="https://www.python.org/downloads/"> Here</a>

### :gear: Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/nishantjoshi-007/DubBuddy.git
   ```
2. **Move in the qreator Folder**
   ```bash
   cd DubBuddy
   ```
   
3. **Install Dependencies**
   - Ensure you have Python 3.11.
   - Install the Python package for virtual environment using pip:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Application**
   ```bash
   fastapi dev main.py
   ```

## :dart: Features
- **Seamless Video and Audio Extraction**: Utilizes yt-dlp to effortlessly extract video and audio from YouTube links for processing.
- **Accurate Audio-to-Text Conversion**: Converts audio to text using OpenAI-Whisper, ensuring high accuracy in transcription.
- **Efficient Text Translation**: Translates text into multiple languages with Argos-Translate, catering to a global audience.
- **High-Quality Text-to-Speech Conversion**: Converts translated text to speech with Coqui TTS, providing clear and natural-sounding audio by using voice cloning to enhance the quality and authenticity of the dubbed audio.
- **Subtitle Generation**: Automatically generates subtitles for the newly translated video, making it accessible to a wider audience.
- **User-Friendly Interface**: Designed for ease of use, making it simple for content creators to produce multilingual videos.

## :wave: Contributing
<img src="https://contrib.rocks/image?repo=Louis3797/awesome-readme-template" /> Contributions to the DubBuddy are always welcome! Whether it's reporting bugs, suggesting new features, or improving the code, your input is valuable. Please feel free to fork this repository, make your changes, and submit a pull request.

## :warning: License
Distributed under the MIT License. See <a href="https://github.com/nishantjoshi-007/DubBuddy/blob/dev/LICENSE">LICENSE</a> for more information.

## :gem: Acknowledgements
This section is used to mention useful resources and libraries that I have used in your projects.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Whisper](https://github.com/openai/whisper)
- [Translation](https://github.com/argosopentech/argos-translate)
- [TTS](https://github.com/coqui-ai/TTS)