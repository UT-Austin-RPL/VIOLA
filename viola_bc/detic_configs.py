
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
import sys
sys.path.insert(0, 'third_party/Detic/third_party/CenterNet2/projects/CenterNet2/')
sys.path.insert(0, 'third_party/Detic/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config


detic_cfg = get_cfg()
add_centernet_config(detic_cfg)
add_detic_config(detic_cfg)
detic_cfg.merge_from_file("third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
# cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
detic_cfg.MODEL.WEIGHTS = 'third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
detic_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
detic_cfg.TEST.DETECTIONS_PER_IMAGE = 7
detic_cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
detic_cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better
# visualization purpose. Set to False for all classes.
def update_path(d):
    for k, v in d.items():
        if isinstance(v, dict):
            update_path(v)
        else:
            if type(v) is str:
                if "datasets/" in v:
                    d[k] = v.replace("datasets/", "third_party/Detic/datasets/")
                    print(d[k])
update_path(detic_cfg)

BUILDIN_CLASSIFIER = {
    'lvis': 'third_party/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}
