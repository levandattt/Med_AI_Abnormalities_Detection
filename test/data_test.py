import pydicom
from pydicom import dcmread
from pydicom.uid import generate_uid
import os

from src.settings import ROOTDIR

dataset = dcmread(
    "/home/xenwithu/Documents/VNPTIT/PythonProject/$1.2.276.0.7230010.3.1.3.2831181056.2775482.1726619429.424604.dicom", force=True)

print (dataset)

filename = f"hihihii.dicom"
print(os.path.join(ROOTDIR, filename))