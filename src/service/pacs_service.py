from pydicom import Dataset
from pydicom.uid import DigitalXRayImageStorageForProcessing, DigitalXRayImageStorageForPresentation
from pynetdicom import AE, debug_logger, evt, build_role
from io import BytesIO
import zipfile

from src.settings import PACS, DEBUG
from src.constraint.uid import PatientStudyOnlyQueryRetrieveInformationModelFind, \
    PatientRootQueryRetrieveInformationModelGet


class PacsService:
    def store_dcm(self, dicom):
        print("STORE CALLED")
        if DEBUG:
            debug_logger()

        ae = AE()

        SOPClassUID = dicom.file_meta.MediaStorageSOPClassUID
        TransferSyntaxUID = dicom.file_meta.TransferSyntaxUID
        ae.add_requested_context(SOPClassUID, TransferSyntaxUID)

        assoc = ae.associate(PACS.get('server'), PACS.get('port'), ae_title=PACS.get('ae_title'))
        if assoc.is_established:
            status = assoc.send_c_store(dicom)
            if status:
                print('C-STORE request status: 0x{0:04x}'.format(status.Status))
            else:
                print('Connect  ion timed out, was aborted or received invalid response')
            assoc.release()
            return status
        else:
            print("Association rejected, aborted or never connected")

    def find_series_by_study(self, patient_id, study_uid):
        # if DEBUG:
        #     debug_logger()

        ae = AE()
        ae.add_requested_context(PatientStudyOnlyQueryRetrieveInformationModelFind)
        assoc = ae.associate(PACS.get('server'), PACS.get('port'), ae_title=PACS.get('ae_title'))

        if assoc.is_established:
            ds = Dataset()
            ds.QueryRetrieveLevel = 'SERIES'
            ds.PatientID = patient_id
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = ''
            ds.SeriesDate=''
            ds.SeriesTime=''

            responses = assoc.send_c_find(ds, PatientStudyOnlyQueryRetrieveInformationModelFind)
            results = []
            for (status, identifier) in responses:
                if status:
                    if status.Status in (0xFF00, 0xFF01):
                        result = {}
                        for tag, element in identifier.items():
                            result[element.keyword] = element.value
                        results.append(result)
                else:
                    print('Connection timed out, was aborted or received invalid response')
            assoc.release()
            return results
        else:
            print('Association rejected, aborted or never connected')



    def find(ds):
        if DEBUG:
            debug_logger()

        ae = AE()
        ae.add_requested_context(PatientStudyOnlyQueryRetrieveInformationModelFind)
        assoc = ae.associate(PACS.get('server'), PACS.get('port'), ae_title=PACS.get('ae_title'))

        if assoc.is_established:
            # Send the C-FIND request
            responses = assoc.send_c_find(ds, PatientStudyOnlyQueryRetrieveInformationModelFind)
            results = []
            for (status, identifier) in responses:
                # print('=====================================')
                if status:
                    # if status.Status == 0xFF00:
                    #     print('C-FIND query status: 0x{0:04X}'.format(status.Status))
                    #     print('-------------------------------------')
                    #     print(identifier)
                    if status.Status in (0xFF00, 0xFF01):
                        result = {}
                        for tag, element in identifier.items():
                            result[element.keyword] = element.value
                        results.append(result)
                # else:
                #     print('Connection timed out, was aborted or received invalid response')
            # Release the association
            assoc.release()
            return results
        else:
            print('Association rejected, aborted or never connected')
    def handle_store(self, event):
        dataset = event.dataset
        dataset.file_meta = event.file_meta

        series_instance_uid = dataset.SeriesInstanceUID
        sop_instance_uid = dataset.SOPInstanceUID
        file_name = f"{series_instance_uid}/{sop_instance_uid}.dcm"

        dicom_bytes = BytesIO()
        dataset.save_as(dicom_bytes, write_like_original=False)
        dicom_bytes.seek(0)  # Đặt lại vị trí đọc trong buffer
        self.zip_file.writestr(file_name, dicom_bytes.read())
        print(f"Added DICOM to ZIP: {file_name}")

        return 0x0000

    def get_studies(self, patient_id, study_uid):

        self.zip_file_buffer = BytesIO()

        self.zip_file = zipfile.ZipFile(self.zip_file_buffer, 'w', zipfile.ZIP_DEFLATED)

        # Implement the handler for evt.EVT_C_STORE
        handlers = [(evt.EVT_C_STORE, self.handle_store)]

        # Initialise the Application Entity
        ae = AE()

        # Add the requested presentation contexts (QR SCU)
        ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
        ae.add_requested_context(DigitalXRayImageStorageForProcessing)
        ae.add_requested_context(DigitalXRayImageStorageForPresentation)
        role_processing = build_role(DigitalXRayImageStorageForProcessing, scp_role=True, scu_role=False)
        role_presentation = build_role(DigitalXRayImageStorageForPresentation, scp_role=True, scu_role=False)

        ds = Dataset()
        ds.QueryRetrieveLevel = 'STUDY'
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid

        assoc = ae.associate(PACS.get('server'), PACS.get('port'), ae_title=PACS.get('ae_title'), ext_neg=[role_processing, role_presentation],
                             evt_handlers=handlers)
        if assoc.is_established:
            responses = assoc.send_c_get(ds, PatientRootQueryRetrieveInformationModelGet)
            for (status, identifier) in responses:
                if status:
                    print('C-GET query status: 0x{0:04x}'.format(status.Status))
                else:
                    print('Connection timed out, was aborted or received invalid response')

            # Đóng zip sau khi hoàn thành
            self.zip_file.close()

            # Lấy dữ liệu ZIP từ buffer
            assoc.release()
            return self.zip_file_buffer.getvalue()
        else:
            print("Association rejected, aborted or never connected")
        return None

    def get_series_handle_store(self, event):
        print("hihihi")
        dataset = event.dataset
        dataset.file_meta = event.file_meta
        self.series_instance = dataset
        return 0x0000

    def get_dicom(self, patient_id, study_uid, series_uid):
        # if DEBUG:
        #     debug_logger()
        handlers = [(evt.EVT_C_STORE, self.get_series_handle_store)]

        ae = AE()
        ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
        ae.add_requested_context(DigitalXRayImageStorageForProcessing)
        ae.add_requested_context(DigitalXRayImageStorageForPresentation)
        role_processing = build_role(DigitalXRayImageStorageForProcessing, scp_role=True, scu_role=False)
        role_presentation = build_role(DigitalXRayImageStorageForPresentation, scp_role=True, scu_role=False)

        assoc = ae.associate(PACS.get('server'), PACS.get('port'), ae_title=PACS.get('ae_title'), ext_neg=[role_processing, role_presentation], evt_handlers=handlers)

        if assoc.is_established:
            ds = Dataset()
            ds.QueryRetrieveLevel = 'SERIES'
            ds.PatientID = patient_id
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid

            responses = assoc.send_c_get(ds, PatientRootQueryRetrieveInformationModelGet)
            for (status, identifier) in responses:
                if status:
                    print('C-GET query status: 0x{0:04x}'.format(status.Status))
                else:
                    print('Connection timed out, was aborted or received invalid response')
            assoc.release()
            if self.series_instance:
                return self.series_instance
        else:
            print('Association rejected, aborted or never connected')
