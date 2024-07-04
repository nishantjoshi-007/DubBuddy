from pydantic import BaseModel, EmailStr
from fastapi import Form, File, UploadFile
from typing import Optional, List

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