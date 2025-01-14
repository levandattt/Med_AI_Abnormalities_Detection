from typing import Dict, Any
from fastapi.responses import StreamingResponse
import json

class DicomJSONResponse(StreamingResponse):
    def __init__(self, dicom_data: bytes, json_data: Dict[Any, Any]):
        # Define boundary for multipart
        self.boundary = "boundary"

        # Create the multipart content
        content = self.create_multipart_content(dicom_data, json_data)

        super().__init__(
            content=content,
            media_type=f"multipart/mixed; boundary={self.boundary}"
        )

    def create_multipart_content(self, dicom_data: bytes, json_data: Dict[Any, Any]):
        # Convert the content to bytes and yield in chunks
        def generate():
            # Start boundary
            yield f"--{self.boundary}\r\n".encode()

            # DICOM part
            yield "Content-Type: application/dicom\r\n".encode()
            yield "Content-Disposition: attachment; filename=image.dcm\r\n\r\n".encode()
            yield dicom_data
            yield "\r\n".encode()

            # JSON part
            yield f"--{self.boundary}\r\n".encode()
            yield "Content-Type: application/json\r\n\r\n".encode()
            yield json.dumps(json_data).encode()
            yield "\r\n".encode()

            # End boundary
            yield f"--{self.boundary}--".encode()

        return generate()