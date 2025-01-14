
import cv2
from pydicom.uid import generate_uid

from numpy import ndarray
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from typing import List
import os
import time
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
from typing import Any, Union
import yaml
import pandas as pd
import tqdm
import pickle
from PIL import Image
import io
import base64
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian

# from full_convmf_vn import ConvMF, Recommender
import numpy as np

from src.settings import ROOTDIR, PREDICT_RESULT_DIR

MEASURE_DOT = "dot product aka. inner product"


def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = yaml.full_load(f)
    return content


print('location: ', os.getcwd())



@dataclass
class Flags:
    # General
    debug: bool = True
    outdir: str = "results/det"

    # Data config
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    split_mode: str = "all_train"  # all_train or valid20
    seed: int = 111
    train_data_type: str = "original"  # original or wbf
    use_class14: bool = False
    # Training config
    iter: int = 10000
    ims_per_batch: int = 2  # images per batch, this corresponds to "total batch size"
    num_workers: int = 4
    lr_scheduler_name: str = "WarmupMultiStepLR"  # WarmupMultiStepLR (default) or WarmupCosineLR
    base_lr: float = 0.00025
    roi_batch_size_per_image: int = 512
    eval_period: int = 10000
    aug_kwargs: Dict = field(default_factory=lambda: {})

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self


def format_pred(labels: ndarray, boxes: ndarray, scores: ndarray) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")
    return " ".join(pred_strings)


def predict_batch(predictor: DefaultPredictor, im_list: List[ndarray]) -> List:
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        for original_image in im_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # Do not apply original augmentation, which is resize.
            # image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)
        return predictions


thing_classes = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis"
]
category_name_to_id = {class_name: index for index, class_name in enumerate(thing_classes)}


def get_vinbigdata_dicts_test(
        imgdir: Path, test_meta: pd.DataFrame, use_cache: bool = True, debug: bool = True,
):
    debug_str = f"_debug{int(debug)}"
    cache_path = Path(".") / f"dataset_dicts_cache_test{debug_str}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        # test_meta = pd.read_csv(imgdir / "test_meta.csv")
        if debug:
            test_meta = test_meta.iloc[:500]  # For debug....

        # Load 1 image to get image size.
        image_id = test_meta.loc[0, "image_id"]
        image_path = str(imgdir / "test" / f"{image_id}.png")
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, test_meta_row in tqdm(test_meta.iterrows(), total=len(test_meta)):
            record = {}

            image_id, height, width = test_meta_row.values
            filename = str(imgdir / "test" / f"{image_id}.png")
            record["file_name"] = filename
            # record["image_id"] = index
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            # objs = []
            # record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts


inputdir = Path("")
traineddir = f"{ROOTDIR}/src/model/Faster_RCNN_R_50/model"
# traineddir = inputdir
print('traineddir: ', traineddir)

# flags = Flags()
flags: Flags = Flags().update(load_yaml(str(os.path.join(ROOTDIR,"src/model/Faster_RCNN_R_50/model/flags.yaml"))))
print("flags", flags)
debug = flags.debug
# flags_dict = dataclasses.asdict(flags)
outdir = Path(flags.outdir)
# os.makedirs(str(outdir), exist_ok=True)

datadir = inputdir / "vinbigdata-chest-xray-abnormalities-detection"
if flags.imgdir_name == "vinbigdata-chest-xray-resized-png-512x512":
    imgdir = inputdir / "vinbigdata"
    print('size test 512')
else:
    imgdir = inputdir / flags.imgdir_name
    print('size test 256')

cfg = get_cfg()
original_output_dir = cfg.OUTPUT_DIR
cfg.OUTPUT_DIR = str(outdir)
print(f"cfg.OUTPUT_DIR {original_output_dir} -> {cfg.OUTPUT_DIR}")

cfg.MODEL.DEVICE = "cpu"

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("vinbigdata_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
cfg.SOLVER.MAX_ITER = flags.iter
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

### --- Inference & Evaluation ---
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# path to the model we just trained
cfg.MODEL.WEIGHTS = str(f"{ROOTDIR}/src/model/Faster_RCNN_R_50/model/model_final.pth")
print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
predictor = DefaultPredictor(cfg)

results_list = []
index = 0
batch_size = 4

def check_pixel_data_encapsulation(dataset):
    """Kiểm tra Pixel Data có được encapsulated hay không"""
    try:
        pixel_data_tag = Tag(0x7FE0, 0x0010)  # Tag của Pixel Data
        if pixel_data_tag in dataset and dataset.file_meta.TransferSyntaxUID.is_compressed:
            return True
        return False
    except AttributeError:
        return False


def predict(dicom_data):
        # DICOM file check
        # print("DICOM file detected.", dicom_data)

        data = apply_voi_lut(dicom_data.pixel_array, dicom_data)
        if dicom_data.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        im = Image.fromarray(data)
        im = im.resize((512, 512), Image.LANCZOS)

        im_array = np.array(im)

        image = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)  # Chuyển đổi sang RGB nếu cần
        is_dicom = True


        single_prediction = predict_batch(predictor, [image])
        prediction = single_prediction[0]
        # print("prediction: ", prediction)

        v = Visualizer(
            image[:, :, ::-1],
            metadata=MetadataCatalog.get("vinbigdata_test"),
            scale=1,
            instance_mode=ColorMode.IMAGE_BW
        )
        output = v.draw_instance_predictions(prediction["instances"].to("cpu"))

        # Chuyển đổi ảnh output thành base64 để hiển thị trong template
        output_image = output.get_image()[:, :, ::-1]

        pil_img = Image.fromarray(output_image)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        encoded_img = base64.b64encode(img_byte_arr.read()).decode('utf-8')

        # save file dicom
        if is_dicom:
            boxes = prediction["instances"].pred_boxes.tensor.cpu().numpy()  # Tọa độ bounding box
            classes = prediction["instances"].pred_classes.cpu().numpy()  # Lớp dự đoán
            scores = prediction["instances"].scores.cpu().numpy()  # Xác suất

            # Nếu resize ảnh, tính lại tọa độ bounding box
            scale_x = dicom_data.Columns / im_array.shape[1]
            scale_y = dicom_data.Rows / im_array.shape[0]
            boxes = boxes * [scale_x, scale_y, scale_x, scale_y]

            # Vẽ bounding box lên ảnh gốc
            image_with_boxes = data.copy()  # Dữ liệu pixel gốc từ DICOM
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                label = f"{thing_classes[cls]} ({score:.2f})"

                # Vẽ bounding box
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 6)

                # Thêm nhãn
                cv2.putText(image_with_boxes, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)

            if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
                # If the original was compressed, we need to change to uncompressed
                dicom_data.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            if dicom_data.BitsAllocated == 16:
                # Chuyển đổi ảnh về 16-bit
                image_with_boxes = image_with_boxes.astype(np.uint16)
                image_with_boxes = (image_with_boxes / 255.0 * (2 ** 16 - 1)).astype(np.uint16)

                # Cập nhật PixelData
                dicom_data.PixelData = image_with_boxes.tobytes()
            else:
                # Nếu là ảnh 8-bit
                dicom_data.PixelData = image_with_boxes.tobytes()

            # Cập nhật Rows, Columns
            dicom_data.Rows, dicom_data.Columns = image_with_boxes.shape[:2]

            dicom_data.Rows, dicom_data.Columns = image_with_boxes.shape[:2]
            dicom_data.SamplesPerPixel = 1 if len(image_with_boxes.shape) == 2 else image_with_boxes.shape[2]
            dicom_data.PhotometricInterpretation = "MONOCHROME2"  # Hoặc RGB nếu ảnh là màu

            # dicom_data.BitsAllocated = 8  # hoặc 16 tùy thuộc vào định dạng ảnh
            # dicom_data.BitsStored = 8  # hoặc 16
            # dicom_data.HighBit = 7 if dicom_data.BitsStored == 8 else 15
            # dicom_data.PixelRepresentation = 0  # 0: unsigned int, 1: signed int

            # dicom_data.BitsAllocated = 16  # hoặc 16 tùy thuộc vào định dạng ảnh
            # dicom_data.BitsStored = 14  # hoặc 16
            # dicom_data.HighBit = 13
            # dicom_data.PixelRepresentation = 0  # 0: unsigned int, 1: signed int

            # Giữ nguyên các thông số từ file ban đầu
            dicom_data.BitsAllocated = dicom_data.BitsAllocated  # Giữ nguyên
            dicom_data.BitsStored = dicom_data.BitsStored  # Giữ nguyên
            dicom_data.HighBit = dicom_data.HighBit  # Giữ nguyên
            dicom_data.PixelRepresentation = dicom_data.PixelRepresentation  # Giữ nguyên (0: unsigned, 1: signed)
            return dicom_data

if __name__ == '__main__':
    dicom_file = "/home/xenwithu/Documents/VNPTIT/PythonProject/src/dicom_files/example_private/ANONYMOUS-929845/05_01.dcm"
    dicom_data = pydicom.dcmread(dicom_file)
    predict(dicom_data)

