import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

import pathlib
absolute_path = pathlib.Path(__file__).parent.absolute()

class HopeObject(MujocoXMLObject):
    def __init__(self, name, obj_name):
        if "@" in obj_name:
            obj_name = obj_name.split("@")[0]
        super().__init__(
            str(absolute_path) + "/" + f"{obj_name}.xml",
            # xml_path_completion("objects/{}.xml".format(obj_name)),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

