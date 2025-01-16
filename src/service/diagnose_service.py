import time
from typing import Optional
from pydicom import Dataset, dcmread, DataElement
from pydicom.uid import generate_uid
from matplotlib import pyplot as plt

from src.constraint.Enum.ModelName import ModelName
from src.constraint.default import c_find_default_params
from src.service.pacs_service import PacsService
from src.model.Faster_RCNN_R_50 import faster_rcnn_r50
import copy
from src.constraint.dicom_custom_tag import ai_tag, ai_predict_date_tag, ai_predict_time_tag

def diagnose(patient_id: str, study_uid: str, series_uid: str, model:Optional[str]=ModelName.FASTER_RCNN_R50) -> Optional[Dataset]:
        pacs_service = PacsService()
        data_dict = c_find_default_params.get("SERIES")
        dataset = Dataset()
        for key, value in data_dict.items():
            setattr(dataset, key, value)
        dataset.PatientID = patient_id
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.QueryRetrieveLevel = "SERIES"
        studies = PacsService.find(dataset)
        # studies = pacs_service.find(patient_id, study_uid)
        if not studies:
            return None

        earliest_series = min(studies, key=lambda x: (x['SeriesDate'], x['SeriesTime']))

        dicom_data = pacs_service.get_dicom(
            earliest_series.get('PatientID'),
            earliest_series.get('StudyInstanceUID'),
            earliest_series.get('SeriesInstanceUID')
        )

        dicom_diagnose = copy.deepcopy(dicom_data)

        result = None
        if model == ModelName.FASTER_RCNN_R50:
            result = faster_rcnn_r50.predict(dicom_diagnose)
        if(result):
            result.SeriesInstanceUID = generate_uid()
            result.SOPInstanceUID = generate_uid()
            result.SeriesDescription = "Predicted"
            result.SeriesDate = time.strftime("%Y%m%d")
            result.SeriesTime = time.strftime("%H%M%S")

            pacs_service.store_dcm(result)
            # result.save_as(f"{ROOTDIR}/src/dicom_files/predict_result/{result.SeriesInstanceUID}_{time.time()}.dcm")
            return result, dicom_data
        return None

if __name__ == "__main__":
    dicom = dcmread("/home/xenwithu/Documents/VNPTIT/VinChestXR/src/dicom_files/example_private/ANONYMOUS-929845/01_01.dcm")
    image = dicom.pixel_array
    plt.imshow(image, cmap='gray')
    plt.show()

