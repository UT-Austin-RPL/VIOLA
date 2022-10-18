"""This is a script to process a original dataset into a dataset after the results of detic."""

"""
When saving the new dataset, we need:
1. embedding for each agentview image
2. name to the eye-in-hand image
3. topk_per_image we used
"""

import os
import argparse
import h5py
import numpy as np
from PIL import Image
import cv2

import init_path
from detectron2.config import get_cfg
import sys
sys.path.insert(0, 'third_party/Detic/')
sys.path.insert(0, 'third_party/Detic/third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from models.detic_configs import detic_cfg, BUILDIN_METADATA_PATH, BUILDIN_CLASSIFIER
from models.detic_models import DefaultPredictor, VisualizationDemo
from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.engine import DefaultPredictor
from detic.modeling.utils import reset_cls_test
from detectron2.utils.visualizer import Visualizer

from detectron2.structures import Boxes, Instances
import imageio
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-objects", type=int, default=7)
    parser.add_argument(
        '--nms',
        type=float,
        default=0.05
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    # cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.merge_from_file("third_party/Detic/configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'third_party/Detic/models/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth'
    cfg.MODEL.META_ARCHITECTURE = 'ProposalNetwork'
    cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 128
    # cfg.MODEL.CENTERNET.INFERENCE_TH = 0.1
    cfg.MODEL.CENTERNET.NMS_TH_TEST = args.nms
    # cfg.MODEL.DEVICE='cpu'
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    # cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.3]
    # cfg.MODEL.RPN.IOU_THRESHOLDS = [0.7, 1.0]
    predictor = DefaultPredictor(cfg)

    def proposals2instances(x):
        x.pred_boxes = x.proposal_boxes
        return x
    metadata = MetadataCatalog.get("proposal")
    metadata.thing_classes = [''] # Change here to try your own vocabularies!
    
    # dataset = h5py.File("./datasets/HammerPlaceDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/SingleKitchenDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/tool_hang_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/ToolUseDomain_proficient_set_randomized/demo_large.hdf5", "r+")
    # dataset = h5py.File("./datasets/ToolUseDomain_proficient_set/demo_20.hdf5", "r+")
    # dataset = h5py.File("./datasets/ToolUseDomain_proficient_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/HopeSortingDomain_proficient_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/HopeSortingDomain_proficient_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/SortCreamCheeseDomain_proficient_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/HopeTwoBinsDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/PutBBQBinDomain_training_set/augmented_demo.hdf5", "r+")
    dataset = h5py.File("./datasets/StackTwoTypesDomain_training_set/augmented_demo.hdf5", "r+")
    
    # dataset = h5py.File("./datasets/BinSortingDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/NutAssemblySquare_dataset/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/ToolHangDomain_proficient_set/augmented_demo.hdf5", "r+")

    # dataset = h5py.File("./datasets/ClutterLift_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/SortBBQSauceDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/PutButterDrawerDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/SortCreamBinDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/SortTwoObjectsDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/PutCreamDrawerDomain_training_set/augmented_demo.hdf5", "r+")
    # dataset = h5py.File("./datasets/RealPlaceBBQDomain_training_set/augmented_demo.hdf5", "r+")

    num_demos = dataset["data"].attrs["num_demos"]
    dataset["data"].attrs["nms"] = args.nms

    centernet_bbox = {}

    topks = [20]
    for i in range(num_demos):
        for top_k in topks:
            centernet_bbox[top_k] = []

        images = dataset["data"][f"demo_{i}"]["obs"]["agentview_rgb"][()]
        print(f"Demo {i}")

        img_h = images[0].shape[0]
        img_w = images[0].shape[1]
        print(img_h, img_w)
        # video_writer = imageio.get_writer(f"videos/centernet_{i}_bbox_{top_k}.mp4", fps=20)
        for image in images:
            outputs = predictor(image)

            proposal_boxes = outputs["proposals"].proposal_boxes
            proposal_boxes = proposal_boxes[proposal_boxes.area() < img_h * img_w / 4]
            proposal_boxes = proposal_boxes[proposal_boxes.area() > 4 * 4]
            for top_k in topks:
                centernet_bbox[top_k].append(proposal_boxes.tensor[:top_k])
                # 
                # centernet_bbox[top_k].append(proposal_boxes[proposal_boxes.area() < ].tensor[:top_k])

                # centernet_scores.append(outputs["proposals"].scores[:top_k])
            bbox_tensor = proposal_boxes[proposal_boxes.area() < img_h * img_w / 4][:top_k].tensor

        for top_k in topks:    
            centernet_bbox[top_k]  = torch.stack(centernet_bbox[top_k], axis=0)
            # print(centernet_bbox[top_k].shape)
            try:
                del dataset[f"data/demo_{i}/obs/centernet_bbox_{top_k}"]
            except:
                print("Something wrong")
                pass
            dataset[f"data/demo_{i}/obs"].create_dataset(f"centernet_bbox_{top_k}", data=centernet_bbox[top_k].detach().cpu().numpy())
            print(f"demo {i}, top {top_k}")
            
        # try:
        #     dataset[f"data/demo_{i}/obs"].create_dataset(f"centernet_bbox_{top_k}", data=centernet_bbox.detach().cpu().numpy())
        #     # dataset[f"data/demo_{i}/obs"].create_dataset("centernet_scores", data=centernet_scores.detach().cpu().numpy())
        # except:
        #     pass
    dataset.close()

if __name__ == "__main__":
    main()
