from fastapi import APIRouter
from pydicom import dcmread
from PIL import Image
import numpy as np
import base64
import io

router = APIRouter()

def dicom_to_base64(dicom_file_path, resize_factor=0.5, quality=80):
    # Đọc tệp DICOM
    dicom = dcmread(dicom_file_path)
    pixel_array = dicom.pixel_array

    # Chuẩn hóa pixel array
    pixel_array_normalized = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)

    # Tạo ảnh từ pixel array
    image = Image.fromarray(pixel_array_normalized)

    # Resize hình ảnh (nếu cần)
    new_width = int(image.width * resize_factor)
    new_height = int(image.height * resize_factor)
    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Chuyển đổi hình ảnh thành Base64 trực tiếp
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)  # Nén JPEG
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return base64_str

@router.get("/api/v1/test")
async def test():
    # Đường dẫn tệp DICOM
    dicom1_path = "/home/xenwithu/Documents/VNPTIT/PythonProject/src/dicom_files/example_private/ANONYMOUS-929845/02_01.dcm"
    dicom2_path = "/home/xenwithu/Documents/VNPTIT/PythonProject/src/dicom_files/example_private/ANONYMOUS-929845/03_01.dcm"

    # Chuyển đổi DICOM sang Base64 với prefix
    dicom1_base64 = "data:image/jpeg;base64," + dicom_to_base64(dicom1_path)
    dicom2_base64 = "data:image/jpeg;base64," + dicom_to_base64(dicom2_path)

    # Trả về Base64 của các ảnh kèm prefix
    return {
        "original": dicom1_base64,
        "result": dicom2_base64
    }
