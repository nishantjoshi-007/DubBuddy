from fastapi import FastAPI, Request, Depends, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .schemas.video_convert import Video
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
async def convert(request: Request, video:Video = Depends(Video.as_form)):
    if not video.tos_check: 
        print(f"Video URL: {video.video_url}")
        print(f"Source Language: {video.from_lang}")
        print(f"Target Language: {video.to_lang}")
        print(f"Agreed to Terms of Service: {video.tos_check}")
        return templates.TemplateResponse(request=request, name="index.html")
    
    # Here you can access the form data as a Pydantic model
    print(f"Video URL: {video.video_url}")
    print(f"Source Language: {video.from_lang}")
    print(f"Target Language: {video.to_lang}")
    print(f"Agreed to Terms of Service: {video.tos_check}")
    return templates.TemplateResponse(
        request=request, name="success.html"
    )
    
@app.get("/contact-us", response_class=HTMLResponse)
async def contact_us_page(request: Request):
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
        ("Downloading video", 2),
        ("Extracting audio", 2),
        ("Converting audio to text", 3),
        ("Translating text", 3),
        ("Converting text to audio", 3),
        ("Merging audio and video", 2)
    ]
    
    for index, (task, duration) in enumerate(tasks):
        await websocket.send_json({"step": index + 1, "total_steps": len(tasks), "message": task})
        time.sleep(duration)
    
    await websocket.send_json({"step": len(tasks), "total_steps": len(tasks), "message": "Completed"})
        