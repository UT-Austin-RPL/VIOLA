import numpy as np
import random

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict, array_to_string, CustomMaterial
import robosuite.utils.transform_utils as T

class ShelfObject(CompositeObject):
    def __init__(self,
                 name,
                 shelf_full_size=(0.15, 0.1, 0.05),
                 shelf_thickness=(0.01),
                 use_texture=True):
        
        self._name = name
        self.width = shelf_full_size[0]
        self.length = shelf_full_size[1]
        self.height = shelf_full_size[2]
        self.thickness = shelf_thickness

        base_args = {
            "total_size": self.length / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        geom_mat = "light_wood_mat"

        shelf_width = 0.0
        shelf_length = 0.06

        edge_width = 0.007
        geom_frictions = (0.005, 0.005, 0.0001)

        solref = (0.02, 1.)

        # add_to_dict(
        #     dic=obj_args,
        #     geom_types="box",
        #     geom_locations=(0., 0., 0.0025),
        #     geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
        #     geom_sizes=np.array([self.width, self.length, self.thickness]),
        #     geom_names=f"body_0",
        #     geom_rgbas=None,
        #     geom_materials=geom_mat,
        #     geom_frictions=(0.0005, 0.005, 0.0001),
        #     # geom_frictions=(0.0, 0.0, 0.0),
        #     solref=solref,
        # )

        # add_to_dict(
        #     dic=obj_args,
        #     geom_types="box",
        #     geom_locations=(0., 0., self.height),
        #     geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
        #     geom_sizes=np.array([self.width, self.length, self.thickness]),
        #     geom_names=f"body_1",
        #     geom_rgbas=None,
        #     geom_materials=geom_mat,
        #     geom_frictions=(0.0005, 0.005, 0.0001),
        #     # geom_frictions=(0.0, 0.0, 0.0),
        #     solref=solref,
        # )

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0., self.thickness / 2, 2 * self.height),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([self.width + self.thickness, self.length + self.thickness, self.thickness]),
            geom_names=f"body_2",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=(0.0005, 0.005, 0.0001),
            # geom_frictions=(0.0, 0.0, 0.0),
            solref=solref,
        )

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, self.length + self.thickness/2, self.height - self.thickness / 2),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([self.width, self.thickness, self.height]),
            geom_names=f"body_3",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=1000)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(-self.width,  self.thickness / 2, self.height - self.thickness),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([self.thickness, self.length + self.thickness, self.height]),
            geom_names=f"body_4",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=1000)

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(self.width, self.thickness / 2, self.height - self.thickness),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, 0])), to="wxyz"),
            geom_sizes=np.array([self.thickness, self.length + self.thickness, self.height]),
            geom_names=f"body_5",
            geom_rgbas=None,
            geom_materials=geom_mat,
            geom_frictions=geom_frictions,
            solref=solref,
            density=1000)

        bottom_site = self.get_site_attrib_template()
        top_site = self.get_site_attrib_template()
        horizontal_site = self.get_site_attrib_template()

        bottom_site.update({
            "name": "bottom",
            "pos": array_to_string(np.array([0., 0., -2 * self.height])),
            "size": "0.005",
            "rgba": "0 0 0 0"
        })

        top_site.update({
            "name": "top",
            "pos": array_to_string(np.array([0., 0., 2 * self.height])),
            "size": "0.005",
            "rgba": "0 0 0 0"
        })

        bottom_site.update({
            "name": "bottom",
            "pos": array_to_string(np.array([0., 0., -2 * self.height])),
            "size": "0.005",
            "rgba": "0 0 0 0"
        })

        obj_args.update(base_args)

        obj_args["sites"] = site_attrs
        # obj_args["joints"] = [{"type": "free", "damping":"0.0005"}]
        obj_args["joints"] = None
        super().__init__(**obj_args)

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }

        steel_scratched_material = CustomMaterial(
            texture="WoodLight",
            tex_name="light_wood_tex",
            mat_name="light_wood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
 
        self.append_material(steel_scratched_material)

    @property
    def bottom_offset(self):
        return np.array([0, 0, -2 * self.height])

    @property
    def top_offset(self):
        return np.array([0, 0, 2 * self.height])
        
    @property
    def horizontal_radius(self):
        return self.length * np.sqrt(2)
