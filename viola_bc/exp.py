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
import init_path
from envs import *
from viola_bc.loss import *
from viola_bc.policy import *
from viola_bc.path_utils import checkpoint_model_dir
import utils.utils as utils
from tqdm import trange


import robomimic.utils.file_utils as FileUtils
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

import wandb

from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = TorchUtils.get_torch_device(try_to_use_cuda=True)


def train(cfg):
    data_path = cfg.data.params.data_file_name

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.algo.obs.modality})
    all_obs_keys = []
    for modality_name, modality_list in cfg.algo.obs.modality.items():
        all_obs_keys += modality_list
    # all_obs_keys = ["agentview_rgb", "gripper_states", "bbox"]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=data_path,
            all_obs_keys=all_obs_keys,
            verbose=True)
    model = eval(cfg.algo.model.name)(cfg.algo.model, shape_meta).to(DEVICE)
    print(model)

    if cfg.algo.train.use_rnn:
        seq_len = cfg.algo.train.rnn_horizon
    else:
        seq_len = 1

    if "filter_key" not in cfg.data.params:
        filter_key = None
    else:
        filter_key = cfg.data.params.filter_key
    
    dataset = SequenceDataset(
                hdf5_path=data_path,
                obs_keys=shape_meta["all_obs_keys"],
                dataset_keys=["actions"],
                load_next_obs=False,
                frame_stack=1,
                seq_length=seq_len,                  # length-10 temporal sequences
                pad_frame_stack=True,
                pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
                get_pad_mask=False,
                goal_mode=None,
                hdf5_cache_mode=cfg.hdf5_cache_mode,          # cache dataset in memory to avoid repeated file i/o
                hdf5_use_swmr=False,
                hdf5_normalize_obs=None,
                filter_by_attribute=filter_key,       # can optionally provide a filter key here
            )

    if cfg.hdf5_cache_mode == "low_dim":
        num_workers = 32
    else:
        num_workers = 0
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.algo.train.batch_size,
        shuffle=True,
        num_workers=num_workers
    )    
    
    data = TensorUtils.to_device(next(iter(dataloader)), DEVICE)

    loss_fn = eval(cfg.algo.loss.fn)(**cfg.algo.loss.loss_kwargs).to(DEVICE)
    optimizer = eval(cfg.algo.optimizer.name)(model.parameters(), lr=cfg.algo.train.lr, **cfg.algo.optimizer.parameters)
    if cfg.algo.scheduler is not None:
        if cfg.algo.scheduler.name is not None:
            scheduler = eval(cfg.algo.scheduler.name)(optimizer,
                                                      **cfg.algo.scheduler.parameters)
        else:
            scheduler = None

    train_range = trange(cfg.algo.train.n_epochs)
    
    model_checkpoint_name = cfg.model_dir.model_checkpoint_name

    best_testing_loss = None

    if cfg.algo.train.grad_clip is None:
        cfg.algo.train.grad_clip = 1.0
    print(cfg.experiment_log)
    print("checkpoints: ", model_checkpoint_name)
    for epoch in train_range:
        model.train()

        training_loss = []
        num_iters = len(dataloader)
        for (idx, data) in enumerate(dataloader):
            data = TensorUtils.to_device(data, DEVICE)
            output = model(data)
            if cfg.algo.train.use_rnn:
                loss = loss_fn(output, data["actions"])
            else:
                loss = loss_fn(output, data["actions"].squeeze(1))
            # loss = loss_fn(output, data["actions"].squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.algo.train.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + idx / num_iters)
            training_loss.append(loss.item())

        training_loss = np.mean(training_loss)
        wandb.log({"training loss": training_loss, "epoch": epoch})
            
        if epoch % 5 == 0:
            model.eval()
            testing_loss = []
            for (idx, data) in enumerate(dataloader):
                data = TensorUtils.to_device(data, DEVICE)
                with torch.no_grad():
                    output = model(data)
                # loss = loss_fn(output, data["actions"].squeeze(1))
                if cfg.algo.train.use_rnn:
                    loss = loss_fn(output, data["actions"])
                else:
                    loss = loss_fn(output, data["actions"].squeeze(1))
                testing_loss.append(loss.item())

            testing_loss = np.mean(testing_loss)
            wandb.log({"testing loss": testing_loss, "epoch": epoch})

            if best_testing_loss is None or best_testing_loss > testing_loss:
                best_testing_loss = testing_loss
                utils.torch_save_model(model, model_checkpoint_name, cfg=cfg)

            train_range.set_description(f"Training loss: {np.round(training_loss, 3)}, Testing loss: {np.round(testing_loss, 3)}")    
    return model
    
@hydra.main(config_path="./configs", config_name="config")
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    pp = pprint.PrettyPrinter(indent=4)
    
    checkpoint_model_dir(cfg)
    pp.pprint(cfg)

    cfg.hostname = socket.gethostname()

    utils.set_manual_seeds(cfg.seed)
     
    # Default mode
    wandb_mode = "online"
    if cfg.flags.debug:
        wandb_mode="disabled"
    wandb.init(project=cfg.wandb_project, config=cfg, mode=wandb_mode)
    wandb.run.name = cfg.hostname + ":" + cfg.model_dir.output_dir

    model = train(cfg)
    wandb.finish()

if __name__ == "__main__":
    main()
    
