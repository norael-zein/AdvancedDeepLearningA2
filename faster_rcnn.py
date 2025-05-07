import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger

setup_logger()

 # ==== 1. Register datasets ====
register_coco_instances("my_train", {}, "dataset/train/_annotations_cleaned.coco.json", "dataset/train")
register_coco_instances("my_valid", {}, "dataset/valid/_annotations_cleaned.coco.json", "dataset/valid")
register_coco_instances("my_test", {}, "dataset/test/_annotations_cleaned.coco.json", "dataset/test")

 # ==== 2. Set training config ====
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("my_train",)
cfg.DATASETS.TEST  = ("my_valid",)
cfg.DATALOADER.NUM_WORKERS = 2

 # Load pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # Initial learning rate
cfg.SOLVER.MAX_ITER = 10000    # Total training steps (adjust according to your dataset)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Number of RoI samples per image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # ← Set your number of classes here!

cfg.OUTPUT_DIR = "./output10000"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

 # ==== 3. Freeze backbone (optional) ====
# cfg.MODEL.BACKBONE.FREEZE_AT = 5  # Do not freeze backbone

 # ==== 4. Start training ====
from detectron2.data import build_detection_train_loader, detection_utils as utils
from detectron2.data import transforms as T

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        def mapper(dataset_dict):
            dataset_dict = dataset_dict.copy()
            image = utils.read_image(dataset_dict["file_name"], format="BGR")
            aug = T.AugmentationList([
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                    T.RandomBrightness(0.8, 1.2),
                    T.RandomContrast(0.8, 1.2),
                    T.RandomSaturation(0.8, 1.2),
                    T.ResizeShortestEdge(short_edge_length=(640, 800), max_size=1333),
                    T.RandomFlip(horizontal=True)
                ])
            aug_input = T.AugInput(image)
            transforms = aug(aug_input)
            image = aug_input.image
            annos = [utils.transform_instance_annotations(obj, transforms, image.shape[:2]) for obj in dataset_dict.get("annotations", [])]
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
            dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
            return dataset_dict
        return build_detection_train_loader(cfg, mapper=mapper)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

 # ==== 5. Evaluate model ====
evaluator = COCOEvaluator("my_valid", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_valid")
print("Running evaluation...")
inference_on_dataset(trainer.model, val_loader, evaluator)

