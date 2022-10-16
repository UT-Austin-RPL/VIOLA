import socket
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import h5py
import numpy as np
import cv2
import os
import argparse
import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
import json
from hydra.experimental import compose, initialize
import pprint
from torch.utils.tensorboard import SummaryWriter
import kornia

from robosuite import load_controller_config
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string
import init_path
from envs import *
from viola_bc.policy import *
from viola_bc.centernet_module import load_centernet_rpn
from viola_bc.path_utils import checkpoint_model_dir

import utils.utils as utils
from tqdm import trange


import robomimic.utils.file_utils as FileUtils
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

import imageio

import wandb

from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

MP_ENABLED = False

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method, Array

DEVICE = TorchUtils.get_torch_device(try_to_use_cuda=True)
torch.cuda.synchronize()

class ObsPreprocessorForEval:
    def __init__(self, cfg, use_rpn=False):
        self.gripper_history = []
        self.use_rpn = use_rpn

        if self.use_rpn:
            if "clean_centernet_bbox_20" in cfg.algo.obs.modality.low_dim:
                self.clean_bbox = True
                print("Cleaning bbox")
                self.rpn = load_centernet_rpn(nms=0.1)
            else:
                self.clean_bbox = False
                self.rpn = load_centernet_rpn(nms=cfg.data.nms)

        self.visual_mapping = {"agentview_rgb": "agentview_image",
                               "eye_in_hand_rgb": "robot0_eye_in_hand_image"}

        if cfg.data.env_name == "tool-hang":
            self.visual_mapping["agentview_rgb"] = "sideview_image"
            
        self.proprio_key_mapping = {"gripper_states": "robot0_gripper_qpos", 
                                    "joint_states": "robot0_joint_pos", 
                                    "ee_states": "robot0_eef_pos"}

        print("remember to reset this")
        
    def reset(self):
        self.gripper_history = []

    def get_obs(self, obs, top_k=20):
        """
        args:
           obs (dict): observation dictionary from Robosuite.
        """
        data = {"obs": {}}
    
        data["obs"]["agentview_rgb"] = utils.process_image_input(torch.from_numpy(np.array(obs[self.visual_mapping["agentview_rgb"]]).transpose(2, 0, 1)).float())
        data["obs"]["eye_in_hand_rgb"] = utils.process_image_input(torch.from_numpy(np.array(obs[self.visual_mapping["eye_in_hand_rgb"]]).transpose(2, 0, 1)).float())

        if self.gripper_history == []:
            for _ in range(5):
                self.gripper_history.append(torch.from_numpy(obs["robot0_gripper_qpos"]))
        self.gripper_history.pop(0)
        self.gripper_history.append(torch.from_numpy(obs["robot0_gripper_qpos"]))
        data["obs"]["gripper_history"] = torch.cat(self.gripper_history, dim=-1).float()

        for proprio_state_key, obs_key in self.proprio_key_mapping.items():
            data["obs"][proprio_state_key] = torch.from_numpy(obs[obs_key]).float()
        
        if self.use_rpn:
            outputs = self.rpn(np.array(obs[self.visual_mapping["agentview_rgb"]]))
            img_w, img_h = obs[self.visual_mapping["agentview_rgb"]].shape[:2]
            self.proposal_boxes = outputs["proposals"].proposal_boxes

            self.proposal_boxes = self.proposal_boxes[self.proposal_boxes.area() < img_w * img_h / 4]
            self.proposal_boxes = self.proposal_boxes[self.proposal_boxes.area() > 4 * 4]
            
            self.bbox_tensor = self.proposal_boxes[:top_k].tensor

            if not self.clean_bbox:
                data["obs"][f"centernet_bbox_{top_k}"] = self.bbox_tensor
            else:
                data["obs"][f"clean_centernet_bbox_{top_k}"] = self.bbox_tensor                
        else:
            data["obs"]["stacked_rgb"] = np.concatenate((obs[self.visual_mapping["agentview_rgb"]],
                                                         obs[self.visual_mapping["eye_in_hand_rgb"]]), axis=-1)
            data["obs"]["stacked_rgb"] = stack_obs_processor(torch.from_numpy(data["obs"]["stacked_rgb"]))            
        return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--state-dir',
        type=str
    )
    parser.add_argument(
        '--visualization',
        action="store_true"
    )
    parser.add_argument(
        '--hostname',
        type=str,
        default="./",
    )
    parser.add_argument(
            '--task-name',
            type=str,
            default=None,
            )   
    parser.add_argument(
            '--eval-horizon',
            type=int,
            default=None,
            )   

    parser.add_argument(
        '--att',
        action="store_true"
            )

    parser.add_argument(
        '--topk',
        default=10,
        type=int
    )

    parser.add_argument(
        '--eval-run',
        default=0,
        type=int
    )    
    
    parser.add_argument(
        '--random-seed',
        default=77,
        type=int
    )    
    
    
    return parser.parse_args()

def eval_loop(cfg, state_dir, video_dir, env_args, model, n_eval, success_arr, eval_run_idx, rank, vis_att=False, top_k=10, eval_horizon=None, random_seed=77):
    domain_name = env_args["domain_name"]
    if domain_name == "single-kitchen":
        cfg.eval.max_steps = 5000
    task_name = cfg.eval.task_name
    env_kwargs = env_args["env_kwargs"]
    
    env = TASK_MAPPING[domain_name](
        exp_name=task_name,
        **env_kwargs,
    )

    if env_kwargs["controller_configs"]["type"] == "OSC_POSE":
        action_dim = 7
    elif env_kwargs["controller_configs"]["type"] == "OSC_POSITION":
        action_dim = 4
    else:
        raise ValueError
    
    eval_range = trange(n_eval)

    # This is just my lucky numbers
    if random_seed == 77:
        utils.set_manual_seeds(rank * random_seed)
    else:
        utils.set_manual_seeds(random_seed)
        
    if rank == 0:
        eval_range = trange(n_eval)
    else:
        eval_range = range(n_eval)

    num_success = 0

    try:
        camera_poses = env.CAMERA_VARIANT_POSES
    except:
        camera_poses = []

    print(cfg.algo.model.name)
    if "CenterNet" in cfg.algo.model.name or "SpatialTemporal" in cfg.algo.model.name:
        use_rpn = True
    else:
        use_rpn = False
    obs_preprocessor_for_eval = ObsPreprocessorForEval(cfg=cfg, use_rpn=use_rpn)
    
    for i in eval_range:
        if rank == 0:
            eval_range.set_description(f"{domain_name} - {task_name} Success rate: {num_success} / {i}")
            eval_range.refresh()

        env.reset()

        if "random" not in task_name and "table" not in task_name:
            with np.load(f"scenes/{domain_name}/{task_name}/{eval_run_idx}_{i + rank * 50}.npz") as scene_file:
                initial_mjstate = scene_file["arr_0"]
        elif "table" in task_name:
            with np.load(f"scenes/{domain_name}/normal/{eval_run_idx}_{i + rank * 50}.npz") as scene_file:
                initial_mjstate = scene_file["arr_0"]            
        else:
            initial_mjstate = env.sim.get_state().flatten()

        model_xml = env.sim.model.get_xml()

        if task_name != "camera-change":
            xml = utils.postprocess_model_xml(model_xml, {})
        else:
            camera_pose = camera_poses[i % len(camera_poses)]
            if cfg.data.env_name != "tool-hang":
                camera_name = "agentview"
            else:
                camera_name = "sideview"
            xml = utils.postprocess_model_xml(model_xml, {camera_name: {"pos": array_to_string(camera_pose[0]), "quat": array_to_string(camera_pose[1])}})

        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_mjstate)
        env.sim.forward()

        for _ in range(5):
            env.step([0.] * (action_dim - 1) + [-1.])

        obs = env._get_observations()

        done = False

        if eval_horizon is None:
            max_steps = cfg.eval.max_steps
        else:
            max_steps = eval_horizon
        steps = 0
        obs_preprocessor_for_eval.reset()
        model.reset()

        record_states = []
        record_imgs = []

        while not done and steps < max_steps:
            record_states.append(env.sim.get_state().flatten())
            steps += 1

            data = obs_preprocessor_for_eval.get_obs(obs, top_k=top_k)

            action = model.get_action(data)


            if use_rpn:
                if "SpatialContext" in cfg.algo.model.name or "SpatialTemporal" in cfg.algo.model.name:
                    att_offset = 2
                else:
                    att_offset = 1

                if "SpatialTemporal" in cfg.algo.model.name:
                    transformer_model = model.transformer
                else:
                    transformer_model = model.decoder.transformer_encoder
                    
                attention_output = transformer_model.attention_output
                proposal_boxes = obs_preprocessor_for_eval.proposal_boxes
                bbox_tensor = obs_preprocessor_for_eval.bbox_tensor
                if vis_att:
                    record_imgs.append(obs["agentview_image"])
                    cv2.imshow("agentview", record_imgs[-1][..., ::-1])
                    cv2.waitKey(10)

                else:
                    pass

            else:
                if cfg.data.env_name != "tool-hang":
                    record_imgs.append(obs["agentview_image"])
                else:
                    record_imgs.append(obs["sideview_image"])                    

                if vis_att:
                    cv2.imshow("agentview", record_imgs[-1][..., ::-1])               
                    cv2.waitKey(10)
                
            obs, reward, done, info = env.step(action)
            done = env._check_success()
            if cfg.eval.visualization:
                img = offscreen_visualization(env, use_eye_in_hand=cfg.algo.use_eye_in_hand)

        if done:
            num_success += 1
            for _ in range(10):
                record_states.append(env.sim.get_state().flatten())

        with  h5py.File(f"{state_dir}/{task_name}_eval_run_{eval_run_idx}_ep_{i}_{done}_rank{rank}.hdf5", "w") as state_file:
            state_file.attrs["env_name"] = cfg.data.env_name
            state_file.attrs["model_file"] = env.sim.model.get_xml()
            state_file.create_dataset("states", data=np.array(record_states))
    success_arr[rank] = num_success
    return num_success


def main():
    args = parse_args()

    state_dir = f"{args.hostname}/{args.state_dir}/record_states"
    os.makedirs(state_dir, exist_ok=True)
    
    with open(os.path.join(f"{args.hostname}/{args.state_dir}", "cfg.json"), "r") as f:
        cfg = json.load(f)

    cfg = EasyDict(cfg)
    if args.visualization:
        cfg.eval.visualization = args.visualization
    data_path = cfg.data.params.data_file_name


    # Change obs  modality for images if we are evaluating baseline bc
    if "CenterNet" not in cfg.algo.model.name and  "SpatialTemporal" not in cfg.algo.model.name   and "OREO" not in cfg.algo.model.name:
        ObsUtils.ImageModality.set_obs_processor(processor=stack_obs_processor)
        ObsUtils.ImageModality.set_obs_unprocessor(unprocessor=stack_obs_unprocessor)
    
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.algo.obs.modality})
    all_obs_keys = []
    for modality_name, modality_list in cfg.algo.obs.modality.items():
        all_obs_keys += modality_list

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=data_path,
            all_obs_keys=all_obs_keys,
            verbose=True)
    model = eval(cfg.algo.model.name)(cfg.algo.model, shape_meta).to(DEVICE)
    model_checkpoint_name = cfg.model_dir.model_checkpoint_name    
    
    if args.hostname != "./":
        model_state_dict, _ = utils.torch_load_model(model_checkpoint_name.replace("./", f"./{args.hostname}/"))
    else:
        model_state_dict, _ = utils.torch_load_model(cfg.model_dir.model_checkpoint_name)
    model.load_state_dict(model_state_dict)
    print(model)
    model.eval()
    if not "use_rnn" in cfg.algo.train or not cfg.algo.train.use_rnn:
        model.decoder.policy_output_head.low_noise_eval = False
    # model.decoder.low_noise_eval = True
    data_path = cfg.data.params.data_file_name

    nms = 0.5
    with h5py.File(data_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        try:
            nms = float(f["data"].attrs["nms"])
        except:
            pass
    print(f"Nms : {nms}")
    cfg.data.nms = nms
    env_args["camera_segmentations"] = None
    # Evaluation loop
    num_successes = []

    video_dir = os.path.join(state_dir, "video")
    os.makedirs(video_dir, exist_ok=True)


    num_procs = 1

    num_eval = int(cfg.eval.n_eval / num_procs)
    processes = []

    video_dir = os.path.join(state_dir, "video")
    os.makedirs(video_dir, exist_ok=True)

    wandb_mode = "disabled"
    if cfg.flags.eval_wandb:
        wandb_mode = "online"

    if args.task_name is None:
        cfg.eval.task_name = "normal"
    else:
        cfg.eval.task_name = args.task_name
    wandb.init(project=cfg.eval.wandb_project, config=cfg, mode=wandb_mode)
    wandb.run.name = "centernet: " + ":" + cfg.eval.task_name + ":" + state_dir
    
    for eval_run_idx in range(3):
        success_arr = Array('i', range(num_procs))

        if MP_ENABLED:
            for rank in range(num_procs):
                p = mp.Process(target=eval_loop, args=(cfg, state_dir, video_dir, env_args, model, num_eval, success_arr, eval_run_idx, rank, args.att, args.topk, args.eval_horizon))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            num_success = np.sum(success_arr[:])
        else:
            num_success = eval_loop(cfg, state_dir, video_dir, env_args, model, num_eval, success_arr, eval_run_idx, 0, args.att, top_k=args.topk, eval_horizon=args.eval_horizon, random_seed=args.random_seed)
    
        num_successes.append(num_success / cfg.eval.n_eval)
        print(f"Total success rate: {num_success} / {cfg.eval.n_eval}")

        print(np.mean(num_successes), np.std(num_successes))
        result = {"num_success": np.mean(num_successes)}
        with open(f"{args.hostname}/{args.state_dir}/result_{cfg.eval.task_name}_{eval_run_idx}.json", "w+") as f:
            json.dump(result, f)
    

if __name__ == "__main__":
    main()
    
