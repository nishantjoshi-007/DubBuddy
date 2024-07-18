from pydantic import BaseModel, HttpUrl, EmailStr
from fastapi import Form, File, UploadFile
from typing import Optional, List


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


        
        
class Contact_Form(BaseModel):
    name : str
    email : EmailStr
    subject : str
    message : str
    uploaded_file: Optional[List[UploadFile]] = None
    
    @classmethod
    def as_form(
        cls,
        name : str = Form(...),
        email : EmailStr = Form(...),
        subject : str = Form(...),
        message : str = Form(...),
        uploaded_file: List[UploadFile] = File(None)
    ) -> "Contact_Form":
        
        return cls(
            name = name,
            email = email,
            subject = subject,
            message = message,
            uploaded_file = uploaded_file
        )