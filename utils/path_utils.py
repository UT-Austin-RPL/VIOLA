import os
from easydict import EasyDict
from pathlib import Path
import utils.utils as utils

def baseline_model_dir(cfg, eval_mode=False, run_idx=None):
    # cfg.folder specify the parent folder of the repo, so that we can
    # flexibly change between the cluster and some local desktop
    folder_path = cfg.folder
    
    suffix_str = ""
    if not cfg.algo.slots.use_warmup:
       suffix_str += "_no_warmup"
    subfolder_path = f"slots_{cfg.algo.slots.num_slots}_dim_{cfg.algo.slots.output_dim}_niter_{cfg.algo.slots.slot_n_iters}_hidden_dim_{cfg.algo.slots.hidden_dim}_lr_{cfg.algo.slots.lr}_batch_{cfg.algo.slots.batch_size}{suffix_str}"
    output_dir = folder_path + f"results/{cfg.data.dataset_name}/{subfolder_path}"

    if not eval_mode:
        experiment_id = 0
        for path in Path(output_dir).glob('run_*'):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split('run_')[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        output_dir += f"/run_{experiment_id:03d}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        # We must specify the run_idx if we want to evaluate
        assert(run_idx is not None)
        output_dir += f"/run_{run_idx:03d}"
    
    model_name =  f"{output_dir}/model"

    summary_writer_name = f"{output_dir}/summary"
    return EasyDict({"output_dir": output_dir,
                     "model_name_prefix": model_name,
                     "summary_writer_name": summary_writer_name})

def get_data_modality_str(data_modality):
    data_modality_str = ""
    for modality_str in data_modality[:-1]:
        # Use this function to verify that the specified modalities
        # work.
        print(modality_str)
        assert(modality_str in ["image", "proprio_gripper", "proprio_joints", "proprio_ee"])
        data_modality_str += modality_str + "_"
    data_modality_str += data_modality[-1]
    return data_modality_str

def bc_model_dir(cfg, eval_mode=False, run_idx=None):
    # cfg.folder specify the parent folder of the repo, so that we can
    # flexibly change between the cluster and some local desktop
    folder_path = cfg.folder

    input_data_modality_str = get_data_modality_str(cfg.algo.data_modality)
    # girpper_joints_str = 

    # if cfg.algo.use_gripper and cfg.algo.use_joints:
    #     gripper_joints_str = "gripper_joints_"
    # elif cfg.algo.use_gripper:
    #     gripper_joints_str = "gripper_joints_"
    # elif cfg.algo.use_gripper:
    #     gripper_joints_str = "joints_"
    # else:
    #     gripper_joints_str = ""

    if cfg.algo.random_affine:
        affine_str = f"_aug_{cfg.algo.affine_translate}"
    else:
        affine_str = ""

    suffix_str = f"{input_data_modality_str}{affine_str}"

    if not cfg.algo.use_warmup:
       suffix_str += "_no_warmup"

    subfolder_path = f"lr_{cfg.algo.lr}_batch_{cfg.algo.batch_size}{suffix_str}"
    output_dir = folder_path + f"results/{cfg.algo.name}/{cfg.data.dataset_name}/{subfolder_path}"

    if not eval_mode:
        experiment_id = 0
        for path in Path(output_dir).glob('run_*'):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split('run_')[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        output_dir += f"/run_{experiment_id:03d}"
        os.makedirs(output_dir, exist_ok=True)
        utils.save_run_cfg(output_dir, cfg)
    else:
        # We must specify the run_idx if we want to evaluate
        assert(run_idx is not None)
        output_dir += f"/run_{run_idx:03d}"

        
    model_name =  f"{output_dir}/model"

    summary_writer_name = f"{output_dir}/summary"
    return EasyDict({"output_dir": output_dir,
                     "model_name_prefix": model_name,
                     "summary_writer_name": summary_writer_name})

def robomimic_bc_model_dir(cfg, eval_mode=False, run_idx=None):
    # cfg.folder specify the parent folder of the repo, so that we can
    # flexibly change between the cluster and some local desktop
    folder_path = cfg.folder


    suffix_str = f""

    subfolder_path = f"lr_{cfg.algo.algo.optim_params.policy.learning_rate.initial}_batch_{cfg.algo.train.batch_size}{suffix_str}"
    output_dir = folder_path + f"results/{cfg.algo.algo_name}/{cfg.data.dataset_name}/{subfolder_path}"

    if not eval_mode:
        experiment_id = 0
        for path in Path(output_dir).glob('run_*'):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split('run_')[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        output_dir += f"/run_{experiment_id:03d}"
        os.makedirs(output_dir, exist_ok=True)
        utils.save_run_cfg(output_dir, cfg)
    else:
        # We must specify the run_idx if we want to evaluate
        assert(run_idx is not None)
        output_dir += f"/run_{run_idx:03d}"

        
    model_name =  f"{output_dir}/model"

    summary_writer_name = f"{output_dir}/summary"
    return EasyDict({"output_dir": output_dir,
                     "model_name_prefix": model_name,
                     "summary_writer_name": summary_writer_name})


def detic_gpt2_model_dir(cfg, eval_mode=False, run_idx=None):
    # cfg.folder specify the parent folder of the repo, so that we can
    # flexibly change between the cluster and some local desktop
    folder_path = cfg.folder

    input_data_modality_str = get_data_modality_str(cfg.algo.data_modality)
    # girpper_joints_str = 

    # if cfg.algo.use_gripper and cfg.algo.use_joints:
    #     gripper_joints_str = "gripper_joints_"
    # elif cfg.algo.use_gripper:
    #     gripper_joints_str = "gripper_joints_"
    # elif cfg.algo.use_gripper:
    #     gripper_joints_str = "joints_"
    # else:
    #     gripper_joints_str = ""

    if cfg.algo.random_affine:
        affine_str = f"_aug_{cfg.algo.affine_translate}"
    else:
        affine_str = ""

    suffix_str = f"{input_data_modality_str}{affine_str}"

    if not cfg.algo.use_warmup:
       suffix_str += "_no_warmup"

    subfolder_path = f"lr_{cfg.algo.lr}_batch_{cfg.algo.batch_size}{suffix_str}"
    output_dir = folder_path + f"results/{cfg.algo.name}/{cfg.data.dataset_name}/{subfolder_path}"

    if not eval_mode:
        experiment_id = 0
        for path in Path(output_dir).glob('run_*'):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split('run_')[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        output_dir += f"/run_{experiment_id:03d}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        # We must specify the run_idx if we want to evaluate
        assert(run_idx is not None)
        output_dir += f"/run_{run_idx:03d}"

    model_name =  f"{output_dir}/model"
    summary_writer_name = f"{output_dir}/summary"
    cfg.model_dir = EasyDict({"output_dir": output_dir,
                              "model_name_prefix": model_name,
                              "summary_writer_name": summary_writer_name})
    if not eval_mode:
        utils.save_run_cfg(output_dir, cfg)
    return cfg.model_dir


def segmentation_exp_model_dir(cfg, eval_mode=False, run_idx=None):
    # cfg.folder specify the parent folder of the repo, so that we can
    # flexibly change between the cluster and some local desktop
    folder_path = cfg.folder

    # input_data_modality_str = get_data_modality_str(cfg.algo.data_modality)

    # if cfg.algo.random_affine:
    #     affine_str = f"_aug_{cfg.algo.affine_translate}"
    # else:
    #     affine_str = ""

    # suffix_str = f"{input_data_modality_str}{affine_str}"

    # if not cfg.algo.use_warmup:
    #    suffix_str += "_no_warmup"

    subfolder_path = f"lr_{cfg.algo.lr}_batch_{cfg.algo.batch_size}"
    output_dir = folder_path + f"results/{cfg.algo.name}/{cfg.data.dataset_name}/{subfolder_path}"

    if not eval_mode:
        experiment_id = 0
        for path in Path(output_dir).glob('run_*'):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split('run_')[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        output_dir += f"/run_{experiment_id:03d}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        # We must specify the run_idx if we want to evaluate
        assert(run_idx is not None)
        output_dir += f"/run_{run_idx:03d}"

    model_name =  f"{output_dir}/model"
    summary_writer_name = f"{output_dir}/summary"
    cfg.model_dir = EasyDict({"output_dir": output_dir,
                              "model_name_prefix": model_name,
                              "summary_writer_name": summary_writer_name})
    if not eval_mode:
        utils.save_run_cfg(output_dir, cfg)
    return cfg.model_dir

