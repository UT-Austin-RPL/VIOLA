"""Augmenting without running simulation"""
import h5py
import sys
import init_path
from envs import *
import json
import torchvision
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-folder',
        type=str,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_folder = args.dataset_folder
    assert(dataset_folder not in ["ToolUseDomain_proficient_set_version_1"])
    # dataset_folder = "ToolUseDomain_proficient_set"
    # dataset_folder = "HopeSortingDomain_proficient_set"
    # dataset_folder = "SortCreamCheeseDomain_proficient_set"

    # dataset_folder = "HopeBinsDomain_training_set"
    # dataset_folder = "NutAssemblySquare_dataset"
    # dataset_folder = "BinSortingDomain_training_set"

    demo_file_name = f"./datasets/{dataset_folder}/demo.hdf5"
    new_demo_file_name = f"./datasets/{dataset_folder}/augmented_demo.hdf5"

    demo_file = h5py.File(demo_file_name, "r")
    env_args = demo_file["data"].attrs["env_args"]
    env_args = json.loads(demo_file["data"].attrs["env_args"])
    domain_name = env_args["domain_name"]
    env_kwargs = env_args["env_kwargs"]


    tasks = []
    task_names = ["normal"]
    for task_name in task_names:
        env = TASK_MAPPING[domain_name](
            exp_name=task_name,
            **env_kwargs,
        )

        _ = env.reset()
        tasks.append(env)

    new_demo_file = h5py.File(new_demo_file_name, "w")
    grp = new_demo_file.create_group("data")
    for key in demo_file["data"].attrs.keys():
        grp.attrs[key] = demo_file["data"].attrs[key]

    num_demos = demo_file["data"].attrs["num_demos"]

    # Previous augmentatiddon
    brightness = 0.3
    contrast = 0.3
    saturation = 0.3
    hue = 0.05


    transforms = []
    transforms.append(torchvision.transforms.ColorJitter(brightness=brightness, 
                                                   contrast=contrast, 
                                                   saturation=saturation, 
                                                   hue=hue))

    transforms = torchvision.transforms.Compose(transforms)
    # Start applying radomization after 20 demos

    random_idx = 10

    for demo_idx in range(num_demos):
        print(demo_idx)
        # video_writer = imageio.get_writer(f"../videos/random_{demo_idx}_{task_names[demo_idx % len(tasks)]}.mp4", fps=60)

        for (env_idx, env) in enumerate(tasks):
            ep_grp = grp.create_group(f"demo_{demo_idx + env_idx * num_demos}")
            for key in demo_file[f"data/demo_{demo_idx}"].attrs.keys():
                ep_grp.attrs[key] = demo_file[f"data/demo_{demo_idx}"].attrs[key]
            agentview_images = []
            eye_in_hand_images = []

            for key in demo_file[f"data/demo_{demo_idx}"].keys():
                if key != "obs":
                    ep_grp.create_dataset(key, data=demo_file[f"data/demo_{demo_idx}/{key}"][()])

            agentview_images = demo_file[f"data/demo_{demo_idx}/obs/agentview_rgb"][()]
            eye_in_hand_images = demo_file[f"data/demo_{demo_idx}/obs/eye_in_hand_rgb"][()]
            # for state in demo_file[f"data/demo_{demo_idx}/states"][()]:
            #     env.sim.reset()
            #     env.sim.set_state_from_flattened(state)
            #     env.sim.forward()
            #     env._update_observables(force=True)
            #     obs = env._get_observations()
            #     agentview_images.append(obs["agentview_image"])
            #     eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
            image_len = len(demo_file[f"data/demo_{demo_idx}/obs/agentview_rgb"])
            obs_grp = ep_grp.create_group("obs")

            agentview_tensor = torch.from_numpy(np.stack(agentview_images, axis=0)).permute(0, 3, 1, 2)
            eye_in_hand_tensor = torch.from_numpy(np.stack(eye_in_hand_images, axis=0)).permute(0, 3, 1, 2)

            if demo_idx % 10 != 0:
                input_tensor = torch.cat((agentview_tensor, eye_in_hand_tensor), dim=0)
                out = transforms(input_tensor)
                out = torch.split(out, [len(agentview_tensor), len(eye_in_hand_tensor)], dim=0)
                assert(len(out[0]) == len(agentview_tensor))
                assert(len(out[1]) == len(eye_in_hand_tensor))

                agentview_images = out[0].detach().permute(0, 2, 3, 1).numpy()
                eye_in_hand_images = out[1].detach().permute(0, 2, 3, 1).numpy()
            else:
                agentview_images = np.stack(agentview_images, axis=0)
                eye_in_hand_images = np.stack(eye_in_hand_images, axis=0)

            # assert(agentview_images.shape[1:] == (128, 128, 3))
            # assert(eye_in_hand_images.shape[1:] == (128, 128, 3))

            if domain_name == "tool-hang":
                img_w, img_h = 240, 240
            elif domain_name == "nut-assembly":
                img_w, img_h = 84, 84
            else:
                img_w, img_h = 128, 128
            assert(agentview_images.shape[1:] == (img_w, img_h, 3))
            assert(eye_in_hand_images.shape[1:] == (img_w, img_h, 3))

            obs_grp.create_dataset("agentview_rgb", data=agentview_images)
            obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_images)
            for key in demo_file[f"data/demo_{demo_idx}/obs"].keys():
                if key in ["gripper_history", "joint_states", "ee_states", "gripper_states"]:
                    obs_grp.create_dataset(key, data=demo_file[f"data/demo_{demo_idx}/obs/{key}"][()])
            print(new_demo_file["data"].keys())


    num_demos = new_demo_file["data"].attrs["num_demos"]
    new_demo_file["data"].attrs["num_demos"] = num_demos
    # Post process to have a history of gripper

    try:
        for i in range(num_demos):
            gripper_states = new_demo_file[f"data/demo_{i}/obs/gripper_states"][()]

            gripper_history = []
            for j in range(len(gripper_states)):
                gripper_state_list = []
                for k in range(j-4, j+1):
                    if k < 0:
                        gripper_state_list += gripper_states[0].tolist()
                    else:
                        gripper_state_list += gripper_states[k].tolist()
                gripper_history.append(gripper_state_list)

            print(torch.tensor(gripper_history).shape)
            new_demo_file[f"data/demo_{i}/obs"].create_dataset("gripper_history", data=gripper_history)
    except:
        print("skipping gripper history")
        pass
    # Post process to have stacked images
    for i in range(num_demos):

        agentview_rgb = new_demo_file["data"][f"demo_{i}"]["obs"]["agentview_rgb"][()]
        eye_in_hand_rgb = new_demo_file["data"][f"demo_{i}"]["obs"]["eye_in_hand_rgb"][()]

        stacked_rgb = np.concatenate((agentview_rgb, eye_in_hand_rgb), axis=-1)
        new_demo_file[f"data/demo_{i}/obs"].create_dataset("stacked_rgb", data=stacked_rgb)

    demo_file.close()
    new_demo_file.close()

if __name__ == "__main__":
    main()
