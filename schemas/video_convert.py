from pydantic import BaseModel, HttpUrl
from fastapi import Form

class Video(BaseModel):
    video_url : HttpUrl
    from_lang : str
    to_lang : str
    tos_check : bool
    
    @classmethod
    def as_form(
        cls,
        video_url : HttpUrl = Form(...),
        from_lang : str = Form(...),
        to_lang : str = Form(...),
        tos_check : bool = Form(...)
    ) -> "Video":
        
        return cls(
            video_url = video_url,
            from_lang = from_lang,
            to_lang = to_lang,
            tos_check = tos_check
        )