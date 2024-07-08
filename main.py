from fastapi import FastAPI, Request, Depends, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .schemas.video_convert import Video
from .schemas.contact_form import Contact_Form
from .src.contact_form import send_data_to_database, process_uploaded_files
from .src import combine
import time

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='Static')
templates = Jinja2Templates(directory='templates')


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )
    
@app.post("/success", response_class=HTMLResponse)
async def convert(request: Request, background_task:BackgroundTasks, video: Video = Depends(Video.as_form)):
    print(video.video_url)
    background_task.add_task(combine.final_method, video.video_url, "process", video.from_lang, video.to_lang, video.tos_check)
    return templates.TemplateResponse(
        request=request, name="success.html"
    )
    
@app.get("/contact-us", response_class=HTMLResponse)
async def contact_us_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="contact-us.html"
    )

@app.post("/contact-us", response_class=HTMLResponse)
async def contact_us_form(request: Request, contact_form:Contact_Form = Depends(Contact_Form.as_form)):    
    file_names = process_uploaded_files(contact_form.email, contact_form.uploaded_file)
    
    send_data_to_database(
        contact_form.name, 
        contact_form.email, 
        contact_form.subject, 
        contact_form.message, 
        file_names
    )

    return templates.TemplateResponse(
        request=request, name="contact-us.html"
    )

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="about.html"
    )
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tasks = [
        ("Downloading video", combine.YotubeDownloader),
        ("Extracting audio", combine.YotubeDownloader),
        ("Converting audio to text", combine.AudioProcess),
        ("Translating text", combine.TranslationProcess),
        ("Converting text to audio", 3),
        ("Merging audio and video", 2)
    ]
    
    for index, (task, duration) in enumerate(tasks):
        await websocket.send_json({"step": index + 1, "total_steps": len(tasks), "message": task})
        time.sleep(duration)
    
    await websocket.send_json({"step": len(tasks), "total_steps": len(tasks), "message": "Completed"})
        