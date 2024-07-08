import requests, os
from fastapi import UploadFile
from typing import List, Optional
from pydantic import EmailStr

# Function to process uploaded files and store them
def process_uploaded_files(email : EmailStr, files : Optional[List[UploadFile]]) -> Optional[List[str]]:
    if not files:
        return None

    file_names = []
    for uploaded_file in files:
        if uploaded_file.filename:
            standard_file_name = f"{email} # {uploaded_file.filename}"
            file_location = f"inquires/{standard_file_name}"
            os.makedirs(os.path.dirname(file_location), exist_ok=True)
            with open(file_location, "wb") as f:
                f.write(uploaded_file.file.read())
            file_names.append(standard_file_name)
    
    return file_names


# Function to send data to Google Apps Script
def send_data_to_database(name, email, subject, message, file_names):
    
    data = {
        "name": name,
        "email": email,
        "subject" : subject, 
        "message" : message,
        "files" : file_names if file_names else []
    }
    
    script_url = "https://script.google.com/macros/s/AKfycbwwsBuBCQcvNY7Lk4YlUpAwUGjGxobByesKAiFUDPcUErSCfxMbaLMZC6tAeg5CoLWT/exec"
    
    try:
        print(f"Sending data to Google Sheets: {data}")
        response = requests.post(script_url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"error sending data to google sheet: {e}")
        return {"result":"error", "message":str(e)}