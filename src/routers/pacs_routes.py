from fastapi import APIRouter, File, UploadFile
from src.controllers import pacs_controller
from typing import  List,Optional

router = APIRouter()

@router.post("/api/v1/pacs")
async def upload_study(files: List[UploadFile] = File(...)):
    return await pacs_controller.upload_study(files)

@router.get("/api/v1/pacs", tags=["PACS"])
def find_studies(queryRetrieveLevel:str = 'STUDY', patientID: Optional[str] = None, studyInstanceUID: Optional[str] = None, seriesInstanceUID: Optional[str] = None):
    return  pacs_controller.find(
        queryRetrieveLevel,
        patientID,
        studyInstanceUID,
        seriesInstanceUID
    )

@router.get("/api/v1/studies/export")
async def get_study(patientID: str, studyInstanceUID: str):
    return await pacs_controller.get_studies(patientID, studyInstanceUID)

@router.get("/api/v1/studies/series")
async def get_series(patientId: str, studyInstanceUID: str, seriesInstanceUID: str):
    print("get_series", patientId, studyInstanceUID, seriesInstanceUID)
    return await pacs_controller.get_series(patientId, studyInstanceUID, seriesInstanceUID)

@router.get("/api/v1/studies/{study_uid}")
async def get_study(study_uid: str):
    return await pacs_controller.get_study(study_uid)




