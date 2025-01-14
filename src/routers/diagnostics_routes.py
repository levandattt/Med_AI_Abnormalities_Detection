from fastapi import APIRouter, File, UploadFile
from src.controllers import pacs_controller
from typing import List, Optional

router = APIRouter()

@router.get("/api/v1/diagnose")
async def diagnose(patientID:str, studyInstanceUID:str, seriesInstanceUID:str):
    return await pacs_controller.diagnose(patientID,studyInstanceUID, seriesInstanceUID)



