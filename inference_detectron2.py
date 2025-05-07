import os
import cv2
import torch
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

setup_logger()

# ==== 1. Register test dataset ====
dataset_name = "my_test"
json_path = "dataset/test/_annotations_cleaned.coco.json"
image_path = "dataset/test"

register_coco_instances(dataset_name, {}, json_path, image_path)

# ==== 2. Load model config ====
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = (dataset_name,)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Set your number of classes
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 3. Run evaluation ====
print("Running evaluation on test set...")
evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, dataset_name)
inference_on_dataset(DefaultPredictor(cfg).model, test_loader, evaluator)

# ==== 4. Visualize and save prediction results ====
print("Running prediction and saving example visualizations...")

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(dataset_name)

output_vis_dir = "output/test_predictions"
os.makedirs(output_vis_dir, exist_ok=True)

for fname in os.listdir(image_path):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(image_path, fname)
        img = cv2.imread(img_path)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_img = out.get_image()[:, :, ::-1]

        out_path = os.path.join(output_vis_dir, fname)
        cv2.imwrite(out_path, result_img)
        print(f"Saved: {out_path}")
