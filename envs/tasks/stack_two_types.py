import numpy as np
from easydict import EasyDict
import random
from envs.objects.hope_object import HopeObject
from envs.objects import PotObject
from envs.objects.serving_object import ServingObject

from envs.base_domain import BaseDomain

from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject, CompositeObject
from robosuite.utils.mjcf_utils import xml_path_completion, add_to_dict, array_to_string, CustomMaterial
import robosuite.utils.transform_utils as T
from copy import deepcopy
from envs.utils import MultiRegionRandomSampler


butter_dict = {
    "xrange": [[0.05, 0.10]],
    "yrange": [[-0.10, 0.10]],        
    "rotation": [-np.pi / 10., np.pi / 10.],    
    "z_offset": 0.01,
    "rotation_axis": "z"
}

cream_cheese_dict = {
    "xrange": [[0.05, 0.10]],
    "yrange": [[-0.10, 0.10]],
    "rotation": [-np.pi / 10., np.pi / 10.],
    "z_offset": 0.01,
    "rotation_axis": "z"    
}

cookies_dict = {
    "xrange": [[0.0, 0.05]],
    "yrange": [[-0.20, 0.20]],    
    "rotation": [-np.pi / 6., np.pi / 6.],    
    "z_offset": 0.01,
    "rotation_axis": "z"    
}

butter_base_dict = {
    "xrange": [[0.22, 0.28]],
    "yrange": [[0.05, 0.10]],        
    "rotation": [-np.pi / 20., np.pi / 20.],    
    "z_offset": 0.01,
    "rotation_axis": "z"
}

cream_cheese_base_dict = {
    "xrange": [[0.22, 0.28]],
    "yrange": [[-0.10, -0.05]],    
    "rotation": [-np.pi / 20., np.pi / 20.],    
    "z_offset": 0.01,
    "rotation_axis": "z"
}

# bin_dict = {
#     # "loc": [0.0, -0.18]
#     # "loc": [0.22, -0.2],
#     "bin_size": (0.10, 0.10, 0.10),
#     "xrange": [[0.10, 0.25], [0.25, 0.25], [0.10, 0.25]],
#     "yrange": [[-0.2, -0.18], [-0.2, 0.2], [0.18, 0.2]]
# }

target_region_dict = {
    "loc": [0.25, 0]
}

PUT_BIN_SPECS = EasyDict({
        "objects": ["cream_cheese", "cream_cheese@base", "butter", "butter@base"],
        "butter": butter_dict,
        "cookies": cookies_dict,
        "cream_cheese": cream_cheese_dict,
        "butter@base": butter_base_dict,
        "cream_cheese@base": cream_cheese_base_dict, 
        "target_region": target_region_dict,
    })

def get_stack_two_types_exp_tasks(exp_name="normal", *args, **kwargs):
    task_specs = PUT_BIN_SPECS
    if exp_name == "normal":
        pass

    elif exp_name == "distracting":
        task_specs["objects"] = ["cream_cheese", "cream_cheese@base", "butter", "butter@base", "cookies"]

    elif exp_name == "placement":
        task_specs["cream_cheese"]["xrange"] = [[0.05, 0.10], [0.05, 0.10]]
        task_specs["cream_cheese"]["yrange"] = [[-0.12, -0.10], [0.10, 0.12]]
        task_specs["butter"]["xrange"] = [[0.05, 0.10], [0.05, 0.10]]
        task_specs["butter"]["yrange"] = [[-0.12, -0.10], [0.10, 0.12]]
        pass

    elif exp_name == "camera-change":
        pass
    return StackTwoTypesDomain(task_specs=task_specs,
                             *args,
                             **kwargs)

CAMERA_VARIANT_POSES =  [
        [[0.456131746834771, 0.0, 1.3503500240372423], [0.6380177736282349, 0.3048497438430786, 0.33484986305236816, 0.6380177736282349]],
        [[0.456131746834771, 0.0, 1.3503500240372423], [0.6380177736282349, 0.3348497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.506131746834771, 0.0, 1.3503500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.456131746834771, 0.0, 1.3003500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.456131746834771, 0.0, 1.4003500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]]]

class StackTwoTypesDomain(BaseDomain):
    def __init__(self, 
                 task_specs=PUT_BIN_SPECS,
                 *args, 
                 **kwargs):

        self.hope_objects = []

        objects = task_specs["objects"]
        for obj in objects:
            self.hope_objects.append(obj.lower())

        self.task_specs = task_specs
        self.CAMERA_VARIANT_POSES = CAMERA_VARIANT_POSES
        # kwargs["table_full_size"] = (0.8, 0.8, 0.05)
        super().__init__(*args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        # loading shelf
        self.fixtures_dict["target_region"] = ServingObject(
            name="target_region",
        )
        
        shelf_pos_x, shelf_pos_y = self.task_specs["target_region"]["loc"]
        
        shelf_object = self.fixtures_dict["target_region"].get_obj(); shelf_object.set("pos", array_to_string((shelf_pos_x, shelf_pos_y, self.table_offset[2] * 0 + 0.005))); shelf_object.set("quat", array_to_string((0.0, 0., 0., 1.)));
        mujoco_arena.table_body.append(shelf_object)


    def _load_objects_in_arena(self, mujoco_arena):

        # loading HOPE objects
        for obj in self.hope_objects:
            self.objects_dict[obj] = HopeObject(
                name=obj,
                obj_name=obj
            )

    def _setup_placement_initializer(self, mujoco_arena):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        for i, obj in enumerate(self.hope_objects):
            self.placement_initializer.append_sampler(
                sampler = MultiRegionRandomSampler(
                    name=f"ObjectSampler-{obj}",
                    mujoco_objects=self.objects_dict[obj],
                    x_ranges=self.task_specs[obj]["xrange"],
                    y_ranges=self.task_specs[obj]["yrange"],
                    rotation=self.task_specs[obj]["rotation"],
                    rotation_axis=self.task_specs[obj]["rotation_axis"],
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=self.task_specs[obj]["z_offset"],
                )
            )

    def _check_success(self):

        cream_pos = self.sim.data.body_xpos[self.obj_body_id["cream_cheese"]]
        cream_base_pos = self.sim.data.body_xpos[self.obj_body_id["cream_cheese@base"]]
        butter_pos = self.sim.data.body_xpos[self.obj_body_id["butter"]]
        butter_base_pos = self.sim.data.body_xpos[self.obj_body_id["butter@base"]]
        

        cream_stacked = (np.abs((cream_pos - cream_base_pos)[:2]) <= 0.03).all() and self.check_contact(self.objects_dict["cream_cheese@base"], self.objects_dict["cream_cheese"])
        butter_stacked = (np.abs((butter_pos - butter_base_pos)[:2]) <= 0.03).all() and self.check_contact(self.objects_dict["butter@base"], self.objects_dict["butter"])

        # table_height = self.model.mujoco_arena.table_offset[2]
        # bbq_pos = self.sim.data.body_xpos[self.obj_body_id["bbq_sauce"]]
        # bin_pos = self.sim.data.body_xpos[self.obj_body_id["bin"]]
        # bbq_in_bin = self.objects_dict["bin"].in_box(bin_pos, bbq_pos)
        # return bbq_in_bin
        return cream_stacked and butter_stacked

    def get_object_names(self):

        obs = self._get_observations()

        object_names = list(self.obj_body_id.keys())
        object_names += ["robot0_eef"]

        return object_names

    def _setup_camera(self, mujoco_arena):

        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.456131746834771, 0.0, 1.3503500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[0.456131746834771, 0.0, 1.3503500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )
