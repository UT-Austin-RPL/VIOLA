import os
import utils.utils as utils
from easydict import EasyDict

def checkpoint_model_dir(cfg):
    folder_path = cfg.folder

    subfolder_path = f"lr_{cfg.algo.train.lr}_batch_{cfg.algo.train.batch_size}_{cfg.algo.model.name}"
    output_dir = os.path.join(folder_path, f"results/{cfg.algo.name}-{cfg.data.dataset_name}/{subfolder_path}")
    output_dir = utils.create_run_model(cfg, output_dir)
    model_checkpoint_name = f"{output_dir}/model.pth"
    cfg.model_dir = EasyDict({"output_dir": output_dir,
                              "model_checkpoint_name": model_checkpoint_name})
    utils.save_run_cfg(output_dir, cfg)