#base python environment
FROM python:3.11

#set working directory
WORKDIR /code

#for tts
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#install opencv (subtitle) and ffmpeg (linux video processing)
RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get update && apt-get install -y ffmpeg

#copy and install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#copy the entire fastapi project directory
COPY ./app /code/app

#set working directory
WORKDIR /code/app

#expose the port
EXPOSE 80

#set a default command to run when the container starts
CMD ["fastapi", "run", "main.py", "--port", "80"]