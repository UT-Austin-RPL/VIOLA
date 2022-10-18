"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""

import os
import h5py
import argparse
import random
import numpy as np
import json

import robosuite
# from robosuite.utils.mjcf_utils import postprocess_model_xml
import xml.etree.ElementTree as ET

import time
import init_path
from envs import *
from robosuite.environments.manipulation.pick_place import PickPlaceCan
from robosuite.environments.manipulation.nut_assembly import NutAssemblySquare
import cv2
from PIL import Image
import robosuite.utils.macros as macros
import robosuite.utils.transform_utils as T
import utils.utils as utils
from robosuite.utils import camera_utils

macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
             "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'"
    ),
    parser.add_argument(
        "--use-actions", 
        action='store_true',
    )
    parser.add_argument(
        "--use-camera-obs", 
        action='store_true',
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default="datasets/",
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        default="training_set",
    )

    parser.add_argument(
        '--no-proprio',
        action='store_true'
    )

    parser.add_argument(
        '--domain-name',
        type=str,
        default="tool-use"
    )

    parser.add_argument(
        '--task-name',
        type=str,
        default="normal"
    )
    
    parser.add_argument(
        '--use-depth',
        action='store_true',
    )

    parser.add_argument(
        '--use-seg',
        action='store_true',
    )
    parser.add_argument(
        '--demo-file',
        default="demo.hdf5"
    )
    
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    # env_name = "BlockStacking"
    env_name = f["data"].attrs["env"]

    env_args = f["data"].attrs["env_info"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    out_parent_dir = f"datasets/{env_name}_{args.dataset_name}"
    os.makedirs(out_parent_dir, exist_ok=True)
    hdf5_path = os.path.join(out_parent_dir, args.demo_file)
    print(hdf5_path)
    h5py_f = h5py.File(hdf5_path, "w")

    grp = h5py_f.create_group("data")

    grp.attrs["env_name"] = env_name

    
    utils.update_env_kwargs(env_kwargs,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
        camera_names=["robot0_eye_in_hand",
                      "agentview",
                      # "frontview",
                      ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations="instance" if args.use_seg else None, 
    )

    env = TASK_MAPPING[args.domain_name](
        exp_name=args.task_name,
        creating_dataset=True,
        **env_kwargs,
    )

    env_args = {"type": 1,
                "env_name": env_name,
                "task_name": args.task_name,
                "domain_name": args.domain_name,
                "env_kwargs": env_kwargs}

    grp.attrs["env_args"] = json.dumps(env_args)
    print(grp.attrs["env_args"])
    total_len = 0
    demos = demos

    gripper_states_list = []
    joint_states_list = []
    ee_states_list = []
    agentview_images_list = []
    eye_in_hand_images_list = []
    robot_states_list = []

    task_id_ep_mapping = {}

    cap_index = 5
        
    for (i, ep) in enumerate(demos):
        print("Playing back random episode... (press ESC to quit)")

        # # select an episode randomly
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        env.reset()

        # if env_name == "PegInHoleEnv":
        #     xml = postprocess_model_xml(model_xml, {"agentview": {"pos": "0.7 0 1.45", "quat": "0.653 0.271 0.271 0.653"}})
        # elif env_name == "MediumPickPlace":
        model_xml = utils.postprocess_model_xml(model_xml, {})
        # model_xml = utils.postprocess_model_xml(model_xml, {"robot0_eye_in_hand": {"pos": "0.05 0.0 0.049999", "quat": "0.0 0.707108 0.707108 0.0"}})            
        # else:
        #     xml = postprocess_model_xml(model_xml, {})

        if not args.use_camera_obs:
            env.viewer.set_camera(0)

        # load the flattened mujoco states
        # states = f["data/{}/states".format(ep)].value
        states = f["data/{}/states".format(ep)][()]
        actions = np.array(f["data/{}/actions".format(ep)][()])

        num_actions = actions.shape[0]

        init_idx = 0
        env.reset_from_xml_string(model_xml)
        env.sim.reset()        
        env.sim.set_state_from_flattened(states[init_idx])
        env.sim.forward()
        model_xml = env.sim.model.get_xml()

        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_image_names = []
        eye_in_hand_image_names = []

        agentview_images = []
        eye_in_hand_images = []

        agentview_depths = []
        eye_in_hand_depths = []

        agentview_seg = {0: [],
                         1: [],
                         2: [],
                         3: [],
                         4: []}
        # os.makedirs(f"{out_parent_dir}/ep_{i}", exist_ok=True)

        idx = 0

        rewards = []
        dones = []
        object_names = env.get_object_names()

        object_states = {}

        for name in object_names:
            object_states[name] = []
        
        for j, action in enumerate(actions):

            obs, reward, done, info = env.step(action)

            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                # assert(np.all(np.equal(states[j + 1], state_playback)))
                err = np.linalg.norm(states[j + 1] - state_playback)

                if err > 0.01:
                    print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

            # Skip recording because the force sensor is not stable in
            # the beginning
            if j < cap_index:
                continue

            for name in object_names:
                object_states[name].append(np.hstack((obs[f"{name}_pos"],
                                                      T.quat2axisangle(obs[f"{name}_quat"])))
                )

            if not args.no_proprio:
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])

                joint_states.append(obs["robot0_joint_pos"])

                if env_kwargs["controller_configs"]["type"] == "OSC_POSITION":
                    ee_states.append(np.hstack((obs["robot0_eef_pos"])))
                else:
                    ee_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))

            robot_states.append(env.get_robot_state_vector(obs))

            if args.use_camera_obs:

                if args.use_depth:
                    agentview_depths.append(obs["agentview_depth"])
                    eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                if args.use_seg:
                    seg_img = obs["agentview_segmentation_instance"]

                    seg_img[seg_img==5] = 0
                    seg_img[seg_img==4] = 0
                    seg_img[seg_img==6] = 4                    
                    # Currently hard coded
                    for idx in range(seg_img.max() + 1):
                        mask_img = (seg_img == idx).astype(np.uint8).squeeze()
                        import cv2
                        mask_img = cv2.medianBlur(mask_img, 5)
                        agentview_seg[idx].append(np.expand_dims(mask_img, axis=-1))
                        # agentview_seg[idx].append(obs["agentview_image"] * (seg_img == idx).astype(np.uint8))

                            
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                # eye_in_hand_image = cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_BGR2RGB)
                # cv2.imwrite(eye_in_hand_image_name, eye_in_hand_image)
                # agentview_image = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_BGR2RGB)
                # cv2.imwrite(agentview_image_name, agentview_image)

                # assert(np.all(obs["agentview_image"] == np.array(Image.open(agentview_image_name))))

                # agentview_images.append(obs["agentview_image"].transpose(2, 0, 1))
                # eye_in_hand_images.append(obs["robot0_eye_in_hand_image"].transpose(2, 0, 1))


            else:
                env.render()

        states = states[cap_index:]
        actions = actions[cap_index:]
        dones = np.zeros(len(actions)).astype(np.uint8)
        dones[-1] = 1
        rewards = np.zeros(len(actions)).astype(np.uint8)
        rewards[-1] = 1
        assert(len(actions) == len(agentview_images))
        print(len(actions))


        ep_data_grp = grp.create_group(f"demo_{i}")

        obs_grp = ep_data_grp.create_group("obs")
        if not args.no_proprio:
            obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
        obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
        obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))

        if args.use_depth:
            obs_grp.create_dataset("agentview_depth", data=np.stack(agentview_depths, axis=0))
            obs_grp.create_dataset("eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0))

        # Currently hard coded
        if args.use_seg:
            for idx in range(5):
                obs_grp.create_dataset(f"agentview_segmentation_{idx}", data=np.stack(agentview_seg[idx], axis=0))
        # Create ground truth object_states
        for (name, state_list) in object_states.items():
            ep_data_grp.create_dataset(name, data=np.stack(state_list, axis=0))
        
        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("states", data=states)
        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)
        ep_data_grp.attrs["num_samples"] = len(agentview_images)
        ep_data_grp.attrs["model_file"] = model_xml
        ep_data_grp.attrs["init_state"] = states[init_idx]
        total_len += len(agentview_images)

    grp.attrs["num_demos"] = len(demos)
    grp.attrs["total"] = total_len
    
    h5py_f.close()        
    f.close()
