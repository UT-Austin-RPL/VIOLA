import os
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, BoxObject, BallObject, CompositeObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material
from robosuite.utils.buffers import RingBuffer

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class BaseDomain(SingleArmEnv):
    """
    A base class for stage 1&2 domains. Currently two domains inherit this BaseDomain class: PickPlaceDomain and PushDomain. They can be found in pick_place_domain.py and push_domain.py.
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        creating_dataset=False,
        table_full_size=(1.0, 0.8, 0.05),
        table_offset=(-0.2, 0, 0.90),
        renderer="mujoco",
        table_xml_file="../assets/normal_table.xml",
        **kwargs
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = table_full_size
        self.table_offset = table_offset

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.objects_dict = {}
        self.fixtures_dict = {}

        self.objects = []
        self.fixtures = []
        self.custom_material_dict = {}

        self.custom_asset_dir = os.path.abspath("./assets")

        self.creating_dataset = creating_dataset

        self.table_xml_file = table_xml_file

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            **kwargs
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the task succeeds.

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_fixtures_in_arena(self, mujoco_arena):
        raise NotImplementedError
    
    def _load_objects_in_arena(self, mujoco_arena):
        raise NotImplementedError

    def _load_distracting_objects(self, mujoco_arena):
        raise NotImplementedError

    def _load_custom_material(self):
        """
        Define all the textures
        """
        self.custom_material_dict = dict()

        tex_attrib = {
            "type": "cube"
        }

        # for texture_name in ["red_table_cloth",
        #                          "blue_table_cloth",
        #                          "yellow_table_cloth",
        #                          "green_table_cloth",
        #                          "dark_table_cloth",
        #                          "wood_table_cloth",
        #                          "kanagawa_wave_table_cloth",
        #                          "matcha"]:
        #     self.custom_material_dict[texture_name] = CLCustomMaterial(
        #         texture=f"Tex_{texture_name}",
        #         tex_name=texture_name,
        #         mat_name=f"Mat_{texture_name}",
        #         texture_file_path=os.path.join(self.custom_asset_dir, f"{texture_name}.png"),
        #         tex_attrib=tex_attrib,
        #         mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        #     )

        self.custom_material_dict["bread"] = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )
        self.custom_material_dict["darkwood"] = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        self.custom_material_dict["lightwood"] = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )
        
        self.custom_material_dict["metal"] = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib=tex_attrib,
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }

        self.custom_material_dict["greenwood"] = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.custom_material_dict["redwood"] = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        self.custom_material_dict["bluewood"] = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="MatBlueWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.custom_material_dict["lemon"] = CustomMaterial(
            texture="Lemon",
            tex_name="lemon",
            mat_name="MatLemon",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.custom_material_dict["steel"] = CustomMaterial(
            texture="SteelScratched",
            tex_name="steel_scratched_tex",
            mat_name="MatSteelScratched",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

    def _setup_camera(self, mujoco_arena):
        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="canonical_agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001),
            xml=os.path.join(FILEPATH, self.table_xml_file),
        )
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self._setup_camera(mujoco_arena)
        
        self._load_custom_material()

        self._load_fixtures_in_arena(mujoco_arena)

        self._load_objects_in_arena(mujoco_arena)

        self._setup_placement_initializer(mujoco_arena)

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())
        for fixture in self.fixtures:
            if issubclass(type(fixture), CompositeObject):
                continue
            for material_name, material in self.custom_material_dict.items():
                tex_element, mat_element, _, used = add_material(root=fixture.worldbody,
                                                                 naming_prefix=fixture.naming_prefix,
                                                                 custom_material=deepcopy(material))
                fixture.asset.append(tex_element)
                fixture.asset.append(mat_element)
        

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.objects,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)
    

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = dict()

        for (object_name, object_body) in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(object_body.root_body)

        for (fixture_name, fixture_body) in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(fixture_body.root_body)
            
        
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True
        
        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            sensors = []
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked drawer version with rotatable handle

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)

        sensors.append(world_pose_in_gripper)
        names.append("world_pose_in_gripper")

        for (i, obj) in enumerate(self.objects + self.fixtures):
            obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality="object")

            sensors += obj_sensors
            names += obj_sensor_names
            
        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq
                )
                
        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names
    
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        """
        Check if drawer has been opened.

        Returns:
            bool: True if drawer has been opened
        """
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4 and len(action) > 4:
            # Convert OSC_POSITION action
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)
        
        obs, reward, done, info = super().step(action)
        done = self._check_success()
        return obs, reward, done, info
        
        
    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # Check if stove is turned on or not
        self._post_process()

        return reward, done, info

    def _post_process(self):
        pass

    def get_robot_state_vector(self, obs):
        return np.concatenate([obs["robot0_gripper_qpos"],
                               obs["robot0_eef_pos"],
                               obs["robot0_eef_quat"]])

    # def reset_to(self, states):
    #     self.sim.set_state_from_flattened(states["states"])
    #     self.sim.forward()
    #     return None

    # def get_state(self):
    #     """
    #     Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
    #     """
    #     xml = self.sim.model.get_xml() # model xml file
    #     state = np.array(self.sim.get_state().flatten()) # simulator state
    #     return dict(model=xml, states=state)    
