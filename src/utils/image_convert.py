from PIL import Image
import numpy as np
import base64
import io

def dicom_to_base64(dicom):

    pixel_array = dicom.pixel_array

    pixel_array_normalized = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)

    image = Image.fromarray(pixel_array_normalized)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return image_base64