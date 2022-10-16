import numpy as np
from easydict import EasyDict
import random
from envs.objects.hope_object import HopeObject
from envs.objects import PotObject
from envs.objects.bin_object import BinObject

from envs.base_domain import BaseDomain

from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject, CompositeObject
from robosuite.utils.mjcf_utils import xml_path_completion, add_to_dict, array_to_string, CustomMaterial
import robosuite.utils.transform_utils as T
from copy import deepcopy
from envs.utils import MultiRegionRandomSampler


butter_dict = {
    "xrange": [[0.15, 0.20]],
    "yrange": [[0.10, 0.15]],    
    "rotation": [-np.pi / 6., np.pi / 6.],    
    "z_offset": 0.01,
    "rotation_axis": "z"
}

cream_cheese_dict = {
    "xrange": [[0.15, 0.20]],
    "yrange": [[0.0, 0.05]],    
    "rotation": [-np.pi / 6., np.pi / 6.],
    "z_offset": 0.01,
    "rotation_axis": "z"    
}

bbq_sauce_dict = {
    "xrange": [[0.0, 0.05], [0.25, 0.30]],
    "yrange": [[-0.10, 0.10], [-0.10, 0.10]],    
    "rotation": [-np.pi/2, np.pi/2],
    "z_offset": 0.03,
    "rotation_axis": "x"    
}

cookies_dict = {
    "xrange": [[0.0, 0.05], [0.25, 0.30]],
    "yrange": [[0.10, 0.20], [0.10, 0.20]],    
    "rotation": [-np.pi / 6., np.pi / 6.],    
    "z_offset": 0.01,
    "rotation_axis": "z"    
}

bin_dict = {
    # "loc": [0.0, -0.18]
    # "loc": [0.22, -0.2],
    "bin_size": (0.15, 0.15, 0.10),
    "xrange": [[0.18, 0.20]],
    "yrange": [[-0.2, -0.18]]
}



BIN_SORTING_SPECS = EasyDict({
        "objects": ["cream_cheese", "bbq_sauce", "butter"],
        "butter": butter_dict,
        "bbq_sauce": bbq_sauce_dict,
        "cookies": cookies_dict,
        "cream_cheese": cream_cheese_dict,
        "bin": bin_dict,    
    })

def get_sort_two_objects_exp_tasks(exp_name="normal", *args, **kwargs):
    task_specs = BIN_SORTING_SPECS
    if exp_name == "normal":
        pass

    elif exp_name == "distracting":
        task_specs["objects"] = ["cream_cheese", "bbq_sauce", "cookies", "butter"]

    elif exp_name == "placement":
        task_specs["butter"]["xrange"] = [[0.20, 0.22], [0.20, 0.22]]
        task_specs["butter"]["yrange"] = [[0.08, 0.10], [0.15, 0.17]]

        task_specs["cream_cheese"]["xrange"] = [[0.20, 0.22], [0.20, 0.22]]
        task_specs["cream_cheese"]["yrange"] = [[-0.02, 0.0], [0.05, 0.07]]
        pass

    elif exp_name == "camera-change":
        pass
    return SortTwoObjectsDomain(task_specs=task_specs,
                             *args,
                             **kwargs)

CAMERA_VARIANT_POSES =  [
        [[0.456131746834771, 0.0, 1.3503500240372423], [0.6380177736282349, 0.3048497438430786, 0.33484986305236816, 0.6380177736282349]],
        [[0.456131746834771, 0.0, 1.3503500240372423], [0.6380177736282349, 0.3348497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.506131746834771, 0.0, 1.3503500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.456131746834771, 0.0, 1.3003500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.456131746834771, 0.0, 1.4003500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]]]

class SortTwoObjectsDomain(BaseDomain):
    def __init__(self, 
                 task_specs=BIN_SORTING_SPECS,
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
        # loading bin
        # self.fixtures_dict["bin"] = BinObject(
        #     name="bin",
        #     bin_size=(0.08, 0.08, 0.12),
        #     transparent_walls=False,
        # )
        
        # bin_pos_x, bin_pos_y = self.task_specs["bin"]["loc"]
        
        # bin_object = self.fixtures_dict["bin"].get_obj(); bin_object.set("pos", array_to_string((bin_pos_x, bin_pos_y, self.table_offset[2] * 0 + 0.07))); bin_object.set("quat", array_to_string((0.0, 0., 0., 1.)));
        # mujoco_arena.table_body.append(bin_object)
        pass

    def _load_objects_in_arena(self, mujoco_arena):

        self.objects_dict["bin"] = BinObject(
            name="bin",
            bin_size=self.task_specs["bin"]["bin_size"],
            transparent_walls=False,
            density=10000.
        )
        # for pot_name in ["sauce_pot"]:
        #     self.objects_dict[pot_name] = PotObject(
        #         name=pot_name,
        #         material=deepcopy(self.custom_material_dict[self.task_specs[pot_name]["material"]]),
        #         pot_size=(0.07, 0.015, 0.035)
        #     )

        # loading HOPE objects
        for obj in self.hope_objects:
            self.objects_dict[obj] = HopeObject(
                name=obj,
                obj_name=obj
            )

    def _setup_placement_initializer(self, mujoco_arena):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
        sampler = MultiRegionRandomSampler(
            name=f"ObjectSampler-bin",
            mujoco_objects=self.objects_dict["bin"],
            x_ranges=self.task_specs["bin"]["xrange"],
            y_ranges=self.task_specs["bin"]["yrange"],
            rotation=(0., 0.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
        
        
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

        table_height = self.model.mujoco_arena.table_offset[2]

        cream_pos = self.sim.data.body_xpos[self.obj_body_id["cream_cheese"]]
        butter_pos = self.sim.data.body_xpos[self.obj_body_id["butter"]]
        
        bin_pos = self.sim.data.body_xpos[self.obj_body_id["bin"]]

        cream_in_bin = self.objects_dict["bin"].in_box(bin_pos, cream_pos)
        butter_in_bin = self.objects_dict["bin"].in_box(bin_pos, butter_pos)
        return cream_in_bin and butter_in_bin
        
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
