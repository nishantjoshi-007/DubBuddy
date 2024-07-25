from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .schemas.model import Video, Contact_Form
from .src.contact_form import send_data_to_database, process_uploaded_files
from .src.main import seperate_thread
import os, json, asyncio, logging
from urllib.parse import quote
import uvicorn

#configured logging file
logging.basicConfig(filename='logs/main.log', level=logging.INFO)

app = FastAPI()
app.mount('/static', StaticFiles(directory='./static'), name='Static')
templates = Jinja2Templates(directory='./templates')

translated_video_download = None

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="about.html"
    )

@app.get("/contact-us", response_class=HTMLResponse)
async def contact_us_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="contact-us.html"
    )

@app.post("/contact-us", response_class=HTMLResponse)
async def contact_us_form(request: Request, contact_form:Contact_Form = Depends(Contact_Form.as_form)):    
    file_names = process_uploaded_files(contact_form.email, "./static/inquires",contact_form.uploaded_file)
    
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
    
    
@app.post("/success", response_class=HTMLResponse)
async def convert(request: Request, background_task:BackgroundTasks, video: Video = Depends(Video.as_form)):
    global translated_video_download, processing_status

    async def save_translated_video_path(translated_video_file):
        global translated_video_download, processing_status
        translated_video_download = translated_video_file
        processing_status = "completed"

    processing_status = "processing"
<<<<<<< HEAD
    background_task.add_task(main, video.video_url, "./static/process_videos", video.from_lang, video.to_lang, video.tos_check, save_translated_video_path)
=======
    background_task.add_task(seperate_thread, video.video_url, "./static/process_videos", video.from_lang, video.to_lang, video.tos_check, save_translated_video_path)
>>>>>>> 65b86faa462a464f8abd61aa44b97970b9aa571c
            
    return templates.TemplateResponse(
        request=request, name="success.html"
    )

@app.get("/download-video", response_class=FileResponse)
async def download_video():
    global translated_video_download
    
    if translated_video_download:
        return FileResponse(translated_video_download, media_type="video/mp4", filename=os.path.basename(translated_video_download))


@app.get("/process-status")
async def process_status():
    global translated_video_download, processing_status
    
    if translated_video_download is not None and processing_status == "completed":
        encoded_video_path = quote(translated_video_download)
        print(f"Processing status: {processing_status}, translated video: {encoded_video_path}")
        return JSONResponse(content={"status": processing_status, "final_video": encoded_video_path})
