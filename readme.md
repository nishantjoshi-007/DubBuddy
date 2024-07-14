from TTS.api import TTS
import wexpect

            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            print("model loaded successfully.")

            #to download the model
            command = "python -c 'from TTS.api import TTS; tts = TTS(\"tts_models/multilingual/multi-dataset/xtts_v2\", gpu=False)'"
            child = wexpect.spawn(command, encoding='utf-8', timeout=120)
            child.expect("Otherwise, I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml")
            child.sendline("y")

            if wav_audio_file:
                print("text to speech started.")
                tts.tts_to_file(
                    text = translated_text,
                    file_path = translated_audio_file,
                    speaker_wav = wav_audio_file,
                    language = self.to_lang_code
                )

### imp github links:
- https://github.com/yt-dlp/yt-dlp
- https://github.com/openai/whisper
- https://github.com/argosopentech/argos-translate
- https://github.com/LibreTranslate/argos-translate-files
- https://github.com/coqui-ai/TTS
- https://github.com/espnet/espnet
- https://github.com/facebookresearch/fairseq

### youtube video links for tries:
- https://youtu.be/VaGfBBVorxo?si=laASdDbpG0qkMRMN
- https://youtu.be/2Q_m-sHZgVg?si=_m1shAV1oBnvVRhO


### other imp links:
- https://uiverse.io/
- https://getbootstrap.com/docs/5.3/getting-started/introduction/
- https://fastapi.tiangolo.com/learn/
- https://docs.coqui.ai/en/dev/index.html
- https://pytorch.org/get-started/locally/
- https://fastapi.tiangolo.com/deployment/docker/



To install Python 3.10.12 on your AWS Ubuntu instance, you can follow these steps:

1. **Update Your Package List:**
   First, ensure your package list is up to date.

   ```bash
   sudo apt update
   ```

2. **Install Prerequisites:**
   Install necessary packages for building Python from source.

   ```bash
   sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
   ```

3. **Download Python 3.10.12 Source Code:**
   Navigate to Python's official website to find the download link for Python 3.10.12 source code. Then, use `wget` to download it directly on your instance.

   ```bash
   wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
   ```

4. **Extract and Compile Python:**
   Extract the downloaded tarball and navigate into the directory.

   ```bash
   tar -xf Python-3.10.12.tgz
   cd Python-3.10.12
   ```

   Configure and compile Python with optimizations.

   ```bash
   ./configure --enable-optimizations
   make -j $(nproc)
   sudo make altinstall
   ```

   The `--enable-optimizations` flag ensures Python is compiled with optimizations, which may improve its performance.

5. **Verify Installation:**
   Check the installed Python version.

   ```bash
   python3.10 --version
   ```

6. **Update Pip:**
   Ensure Pip is up to date for Python 3.10.

   ```bash
   python3.10 -m pip install --upgrade pip
   ```

7. **Cleanup:**
   Optionally, you can clean up the downloaded source files.

   ```bash
   cd ..
   rm -rf Python-3.10.12
   ```

Now, Python 3.10.12 should be installed on your AWS Ubuntu instance. You can proceed to use this Python version for setting up your NeMo toolkit or any other development tasks.


To create a virtual environment using Python 3.10 on your AWS Ubuntu instance, you can use the `venv` module, which is included in Python 3 by default. Here are the steps:

1. **Navigate to Your Project Directory:**
   Go to the directory where you want to create your virtual environment.

   ```bash
   cd /path/to/your/project
   ```

2. **Create the Virtual Environment:**
   Use Python 3.10 to create a virtual environment named `myenv` (you can replace `myenv` with your preferred name).

   ```bash
   python3.10 -m venv myenv
   ```

   This command will create a new directory `myenv` which contains the virtual environment.

3. **Activate the Virtual Environment:**
   Activate the virtual environment to start using Python 3.10 and its associated Pip.

   ```bash
   source myenv/bin/activate
   devenv\Scripts\activate.bat
   ```

   After activation, your shell prompt will change to indicate that you are now working inside the virtual environment (`(myenv)` typically appears at the beginning of the prompt).

4. **Install Packages:**
   Now, you can install packages using Pip as usual. For example:

   ```bash
   pip install nemo_toolkit[all]
   ```

   This command will install the NeMo toolkit and all its dependencies into your virtual environment.

5. **Deactivate the Virtual Environment:**
   When you're done working in the virtual environment, you can deactivate it.

   ```bash
   deactivate
   ```

   This returns you to your regular shell environment.

That's it! You now have Python 3.10 installed in a virtual environment on your AWS Ubuntu instance, ready to use for your development tasks.