from easydict import EasyDict
import numpy as np
import random
from copy import deepcopy
from envs.base_domain import BaseDomain
from robosuite.models.objects import CylinderObject, BoxObject
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.mjcf_utils import array_to_string

from envs.objects import PotObject, TargetObject, ButtonObject, StoveObject, SingleKitchenPotObject
from envs.utils import MultiRegionRandomSampler

from robosuite.models.arenas import TableArena
from robosuite.wrappers import DomainRandomizationWrapper


POT_DICT = {"material": "steel",
            "xrange": [[0.175, 0.18]],
            "yrange": [[-0.15, -0.13]]}

BREAD_DICT = {"material": "bread",
              "size": [0.015, 0.025, 0.02],
              "xrange": [[0.05, 0.08]],
              "yrange": [[-0.18, -0.12]]}

STOVE_DICT = {"x": 0.23,
              "y": 0.095}

BUTTON_DICT = {"x": 0.06,
               "y": 0.10}

TARGET_DICT = {"x": 0.345,
                "y": -0.15}

DISTRACT_DICT = {"num": 0,
                 "material_list": ['lightwood', 'darkwood', 'redwood'],
                 "xrange": [[0.40, 0.46], [0.35, 0.46]],
                 "yrange": [[0.05, 0.24], [0.20, 0.24]]}

SINGLE_KITCHEN_SPECS = EasyDict({"pot": POT_DICT,
                           "bread": BREAD_DICT,
                           "stove": STOVE_DICT,
                           "button": BUTTON_DICT,
                           "target": TARGET_DICT,
                           "distracting_objects": DISTRACT_DICT})

def get_single_kitchen_exp_tasks(exp_name="normal", *args, **kwargs):
    task_specs = SINGLE_KITCHEN_SPECS
    if exp_name == "normal":
        pass
    elif exp_name == "placement":
        task_specs["bread"]["xrange"] = [[0.05, 0.08], [0.05, 0.08]]
        task_specs["bread"]["yrange"] = [[-0.20, -0.18], [-0.12, -0.10]]

        task_specs["pot"]["xrange"] = [[0.175, 0.18]]
        task_specs["pot"]["yrange"] = [[-0.18, -0.15]]
    elif exp_name == "distracting":
        task_specs["distracting_objects"]["num"] = 4
        task_specs["distracting_objects"]["material_list"] = ["lightwood", "darkwood"]
    elif exp_name == "camera-change":
        pass

    else:
            raise ValueError
    return SingleKitchenDomain(task_specs=task_specs,
                         *args,
                         **kwargs)

CAMERA_VARIANT_POSES =  [
        [[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423], [0.6380177736282349, 0.3048497438430786, 0.33484986305236816, 0.6380177736282349]],
        [[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423], [0.6380177736282349, 0.3348497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.5886131746834771, -4.392035683362857e-09, 1.4903500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.5386131746834771, -4.392035683362857e-09, 1.5403500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]],
        [[0.5386131746834771, -4.392035683362857e-09, 1.4403500240372423], [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]]]


class SingleKitchenDomain(BaseDomain):
    def __init__(
            self,
            task_specs=SINGLE_KITCHEN_SPECS,
            *args,
            **kwargs):
        self.task_specs = task_specs
        self.CAMERA_VARIANT_POSES = CAMERA_VARIANT_POSES        
        kwargs["table_full_size"] = (1.0, 0.8, 0.05)
        super().__init__(*args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """In Pick Place domain, we do not have environment fixtures"""
        self.fixtures_dict["target"] = TargetObject(
            name="target")
        target_x, target_y = self.task_specs["target"]["x"], self.task_specs["target"]["y"]
        target_object = self.fixtures_dict["target"].get_obj(); target_object.set("pos", array_to_string((target_x, target_y, 0.003))); mujoco_arena.table_body.append(target_object)

        self.fixtures_dict["button"] = ButtonObject(
            name="button")
        button_x, button_y = self.task_specs["button"]["x"], self.task_specs["button"]["y"]
        button_object = self.fixtures_dict["button"].get_obj(); button_object.set("quat", array_to_string((0., 0., 0., 1.))); button_object.set("pos", array_to_string((button_x, button_y, 0.02))); mujoco_arena.table_body.append(button_object)

        self.fixtures_dict["stove"] = StoveObject(
            name="stove")
        stove_x, stove_y = self.task_specs["stove"]["x"], self.task_specs["stove"]["y"]
        stove_object = self.fixtures_dict["stove"].get_obj(); stove_object.set("pos", array_to_string((stove_x, stove_y, 0.02))); mujoco_arena.table_body.append(stove_object)
        
        
    def _load_objects_in_arena(self, mujoco_arena):
        # Load environment fixtures

        # self.objects_dict["pot"] = PotObject(
        #     name="pot",
        #     material=deepcopy(self.custom_material_dict[self.task_specs["pot"]["material"]])
        # )
        self.objects_dict["pot"] = SingleKitchenPotObject(
            name="pot",
            # material=deepcopy(self.custom_material_dict[self.task_specs["pot"]["material"]])
        )

        bread_size = self.task_specs["bread"]["size"]
        self.objects_dict["bread"] = BoxObject(
            name="bread",
            size_min=bread_size,
            size_max=bread_size,
            rgba=[1, 0, 0, 1],
            material=self.custom_material_dict[self.task_specs["bread"]["material"]],
            density=500.,
        )
        

        if self.task_specs["distracting_objects"]["num"] > 0:
            for i in range(self.task_specs["distracting_objects"]["num"]):
                random_material_name = random.choice(self.task_specs["distracting_objects"]["material_list"])
                self.objects_dict[f"distracting_object_{i}"] = BoxObject(
                    name=f"distracting_object_{i}",
                    size=[0.02, 0.02, 0.02],
                    rgba=[1, 0, 0, 1],
                    material=self.custom_material_dict[random_material_name],
                    density=500.,
                )

    def _setup_placement_initializer(self, mujoco_arena):
        """Function to define the placement"""
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
        sampler = MultiRegionRandomSampler(
            name="ObjectSampler-bread",
            mujoco_objects=self.objects_dict["bread"],
            x_ranges=self.task_specs["bread"]["xrange"],
            y_ranges=self.task_specs["bread"]["yrange"],
            rotation=(-np.pi / 2., -np.pi / 2.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler = MultiRegionRandomSampler(
            name="ObjectSampler-pot",
            mujoco_objects=self.objects_dict["pot"],
            x_ranges=self.task_specs["pot"]["xrange"],
            y_ranges=self.task_specs["pot"]["yrange"],
            rotation=(-0.1, 0.1),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
        
        # self.placement_initializer.append_sampler(
        # sampler = MultiRegionRandomSampler(
        #     name="ObjectSampler-lshape",
        #     mujoco_objects=self.objects_dict["tool"],
        #     x_ranges=self.task_specs["tool"]["xrange"],
        #     y_ranges=self.task_specs["tool"]["yrange"],
        #     rotation=(0., 0.),
        #     rotation_axis='z',
        #     ensure_object_boundary_in_range=False,
        #     ensure_valid_placement=True,
        #     reference_pos=self.table_offset,
        #     z_offset=0.02,
        # ))
        
        # pot_pos_x, pot_pos_y = self.task_specs["pot"]["loc"]
        # pot_object = self.objects_dict["pot"].get_obj(); pot_object.set("pos", array_to_string((pot_pos_x, pot_pos_y, self.table_offset[2] + 0.05)))

        if self.task_specs["distracting_objects"]["num"] > 0:
            for i in range(self.task_specs["distracting_objects"]["num"]):
                distracting_object_name = f"distracting_object_{i}"
                self.placement_initializer.append_sampler(
                    sampler=MultiRegionRandomSampler(
                    name=distracting_object_name,
                    mujoco_objects=self.objects_dict[distracting_object_name],
                    x_ranges=self.task_specs["distracting_objects"]["xrange"],
                    y_ranges=self.task_specs["distracting_objects"]["yrange"],
                    rotation=(-np.pi / 2., np.pi / 2.),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=0.1,
                    )
                )

    def _check_success(self):
        """
        Check if the food item is in the pot
        """
        pot_pos = self.sim.data.body_xpos[self.obj_body_id["pot"]]
        bread_pos = self.sim.data.body_xpos[self.obj_body_id["bread"]]
        target_pos = self.sim.data.body_xpos[self.obj_body_id["target"]]
        dist_target_pot = target_pos - pot_pos
        pot_in_target_region = np.abs(dist_target_pot[0]) < 0.05 and np.abs(dist_target_pot[1]) < 0.10 and np.abs(dist_target_pot[2]) < 0.05
        stove_turned_off = not self.button_on
        if not stove_turned_off:
            self.has_stove_turned_on = True

        object_in_pot = self.check_contact(self.objects_dict["bread"], self.objects_dict["pot"])
        return pot_in_target_region and stove_turned_off and object_in_pot and self.has_stove_turned_on


    def get_object_names(self):
        
        obs = self._get_observations()

        object_names = list(self.obj_body_id.keys())
        object_names += ["robot0_eef"]

        return object_names

    def _reset_internal(self):
        super()._reset_internal()
        self.button_on = False
        self.has_stove_turned_on = False       

    def _setup_references(self):
        super()._setup_references()
        self.button_qpos_addrs = self.sim.model.get_joint_qpos_addr(self.fixtures_dict["button"].joints[0])
        self.sim.data.set_joint_qpos(self.fixtures_dict["button"].joints[0], np.array([-0.4]))        

    def _post_process(self):
        if self.button_on:
            if self.sim.data.qpos[self.button_qpos_addrs] < 0.0:
                self.button_on = False
        else:
            if self.sim.data.qpos[self.button_qpos_addrs] >= 0.0:
                self.button_on = True
        self.fixtures_dict["stove"].set_sites_visibility(sim=self.sim, visible=self.button_on)
