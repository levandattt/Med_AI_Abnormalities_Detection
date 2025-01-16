from fastapi import APIRouter, File, UploadFile
from src.controllers import pacs_controller
from typing import  List,Optional
from fastapi.responses import Response
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

@router.get("/api/v1/studies/series/jpeg")
async def get_series(patientID: str, studyInstanceUID: str, seriesInstanceUID: str):
    print("get_series", patientID, studyInstanceUID, seriesInstanceUID)
    return await pacs_controller.get_series_jpeg(patientID, studyInstanceUID, seriesInstanceUID)


@router.get("/api/v1/studies/{study_uid}")
async def get_study(study_uid: str):
    return await pacs_controller.get_study(study_uid)


# Tạo route cung cấp file XML
@router.get("/xml")
async def get_xml(patientId: str, studyUID: str, seriesUID: str, seriesUID2: str):
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.weasis.org/xsd/2.5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <arcQuery arcId="1000" baseUrl="http://localhost:6969/api/v1/studies/series">
        <Patient PatientID="{patientId}">
            <Study StudyInstanceUID="{studyUID}">
                <Series SeriesInstanceUID="{seriesUID}" SeriesNumber="1">
                    <Instance DirectDownloadFile="?patientId={patientId}&amp;studyInstanceUID={studyUID}&amp;seriesInstanceUID={seriesUID}" InstanceNumber="1" SOPInstanceUID="1.2.840.113619.2.176.2025.1499492.7022.1172755835.87"/>
                </Series>
                <Series SeriesInstanceUID="{seriesUID2}" SeriesNumber="2">
                    <Instance DirectDownloadFile="?patientId={patientId}&amp;studyInstanceUID={studyUID}&amp;seriesInstanceUID={seriesUID2}" InstanceNumber="1" SOPInstanceUID="1.2.840.113619.2.176.2025.1499492.7022.1172755835.87"/>
                </Series>
            </Study>
        </Patient>
    </arcQuery>
</manifest>"""

    # Trả về nội dung XML với Content-Type: application/xml
    return Response(content=xml_content, media_type="application/xml")




