from datetime import datetime

from pydicom import dcmread
from pydicom.uid import generate_uid
import time

from src.constraint.Enum.QueryRetrieveLevel import QueryRetrieveLevel as QueryRetrieveLevelENUM
from src.constraint.default import c_find_default_params
from src.payload.base_res import create_response
from pydicom.dataset import Dataset
from fastapi import UploadFile, Response
from typing import List, Optional
import io
from src.service import diagnose_service
from src.service.pacs_service import PacsService
from src.utils.image_convert import dicom_to_base64


def find(QueryRetrieveLevel:str = "STUDY", PatientID: Optional[str] = '', StudyInstanceUID: Optional[str] = '', SeriesInstanceUID: Optional[str] = ''):
    data_dict = c_find_default_params.get(QueryRetrieveLevel)
    print("data_dict: ", QueryRetrieveLevel)
    dataset = Dataset()
    for key, value in data_dict.items():
        setattr(dataset, key, value)
    dataset.PatientID = PatientID
    dataset.StudyInstanceUID = StudyInstanceUID
    dataset.SeriesInstanceUID = SeriesInstanceUID
    dataset.QueryRetrieveLevel = QueryRetrieveLevel

    results = PacsService.find(dataset)
    if not results:
        return create_response(status="success", data=[], message="No items found")
    return create_response(status="success", data=results, message="Items retrieved successfully")

async def upload_study(files: List[UploadFile]):
    uploaded_file_names = []
    for file in files:

        content = await file.read()
        dicom = dcmread(io.BytesIO(content), force=True)
        if not hasattr(dicom, 'PatientID'):
            dicom.PatientID = f"ANONYMOUS_{time.time()}"
        if not hasattr(dicom, 'PatientName'):
            dicom.PatientName = "Anonymous"
        if not hasattr(dicom, 'StudyInstanceUID'):
            dicom.StudyInstanceUID = generate_uid()
        if not hasattr(dicom, 'SeriesInstanceUID'):
            dicom.SeriesInstanceUID = generate_uid()
        if not hasattr(dicom, 'SOPInstanceUID'):
            dicom.SOPInstanceUID = generate_uid()
        if not hasattr(dicom, 'SOPClassUID'):
            dicom.SOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'
        # ds = pydicom.dcmread(BytesIO(content), force=True)
        # print (dicom)

        # Gửi tệp DICOM đến PACS
        pacs_service = PacsService()
        status = pacs_service.store_dcm(dicom)

        # Kiểm tra trạng thái và xử lý kết quả
        if status:
            print(f"Stored file: {file.filename}")
            uploaded_file_names.append(file.filename)

            print ("status: ",status)
        else:
            print(f"Failed to store file: {file.filename}")


    return {"message": "Files uploaded successfully", "files": uploaded_file_names}

async def get_study(study_uid: str):
    dataset = None
    if (study_uid=="1"):
        dataset = dcmread("/home/xenwithu/Documents/VNPTIT/VinChestXR/src/dicom_files/example_private/ANONYMOUS-929845/02_01.dcm")
    else:
        dataset = dcmread("/home/xenwithu/Documents/VNPTIT/VinChestXR/src/dicom_files/example_private/ANONYMOUS-929845/12_01.dcm", force=True)

    # Ghi Dataset vào một buffer thay vì file
    dicom_buffer = io.BytesIO()
    dataset.save_as(dicom_buffer, write_like_original=False)
    dicom_buffer.seek(0)

    # Trả dữ liệu dưới dạng DICOM file qua HTTP
    return Response(
        dicom_buffer.getvalue(),
        media_type="application/dicom",
        headers={
            "Content-Disposition": f"attachment; filename=1.2.826.0.1.3680043.9.6965.1434327336.483094607.1322137624.dcm"
        }
    )

async def get_studies(patient_id: str, study_uid: str):
    # Tìm kiếm các hình ảnh DICOM từ PACS
    pacs_service = PacsService()
    data = pacs_service.get_studies(patient_id, study_uid)

    if not data:
        return {"message": "No studies found", "data": []}
    return Response(
        data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={study_uid}.zip"
        }
    )

async def diagnose(patient_id:str, study_uid: str, series_uid: str):
    result, original = diagnose_service.diagnose(patient_id, study_uid, series_uid)

    dicom1_base64 = dicom_to_base64(original)
    dicom2_base64 = dicom_to_base64(result)

    return {
        "status": "success",
        "data": {
            "dicomOriginal":{
                "image":"data:image/jpeg;base64,"+dicom1_base64,
                "metadata": {
                    "PatientID": original.PatientID,
                    "StudyInstanceUID": original.StudyInstanceUID,
                    "SeriesInstanceUID": original.SeriesInstanceUID
                }
            },
            "dicomPredict": {
                "image":"data:image/jpeg;base64,"+dicom2_base64,
                "metadata": {
                    "PatientID": result.PatientID,
                    "StudyInstanceUID": result.StudyInstanceUID,
                    "SeriesInstanceUID": result.SeriesInstanceUID
                }
            }
        },
        "message": "Predicted successfully"

    }

async def get_series(patientId: str, studyUID: str, seriesUID: str):
    pacs_service = PacsService()
    dicom_data = pacs_service.get_dicom(patientId, studyUID, seriesUID)

    # Ghi Dataset vào một buffer thay vì file
    dicom_buffer = io.BytesIO()
    dicom_data.save_as(dicom_buffer, write_like_original=False)
    dicom_buffer.seek(0)

    # Trả dữ liệu dưới dạng DICOM file qua HTTP
    return Response(
        dicom_buffer.getvalue(),
        media_type="application/dicom",
        headers={
            "Content-Disposition": f"attachment; filename={seriesUID}.dcm"
        }
    )