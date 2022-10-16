import numpy as np

from robosuite.models.objects import MujocoGeneratedObject, MujocoObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, new_joint, new_inertial,\
    array_to_string, find_elements, add_prefix, OBJECT_COLLISION_COLOR, CustomMaterial

from copy import deepcopy



class OfflineCompositeObject(MujocoGeneratedObject):
    """
    An object constructed out of basic geoms to make more intricate shapes.

    Note that by default, specifying None for a specific geom element will usually set a value to the mujoco defaults.

    Args:
        name (str): Name of overall object

        total_size (list): (x, y, z) half-size in each dimension for the bounding box for
            this Composite object

        geom_types (list): list of geom types in the composite. Must correspond
            to MuJoCo geom primitives, such as "box" or "capsule".

        geom_locations (list): list of geom locations in the composite. Each
            location should be a list or tuple of 3 elements and all
            locations are relative to the lower left corner of the total box
            (e.g. (0, 0, 0) corresponds to this corner).

        geom_sizes (list): list of geom sizes ordered the same as @geom_locations

        geom_quats (None or list): list of (w, x, y, z) quaternions for each geom.

        geom_names (None or list): list of geom names ordered the same as @geom_locations. The
            names will get appended with an underscore to the passed name in @get_collision
            and @get_visual

        geom_rgbas (None or list): list of geom colors ordered the same as @geom_locations. If
            passed as an argument, @rgba is ignored.

        geom_materials (None or list of CustomTexture): list of custom textures to use for this object material

        geom_frictions (None or list): list of geom frictions to use for each geom.

        rgba (None or list): (r, g, b, a) default values to use if geom-specific @geom_rgbas isn't specified for a given element

        density (float or list of float): either single value to use for all geom densities or geom-specific values

        solref (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        solimp (list or list of list): parameters used for the mujoco contact solver. Can be single set of values or
            element-specific values. See http://www.mujoco.org/book/modeling.html#CSolver for details.

        locations_relative_to_center (bool): If true, @geom_locations will be considered relative to the center of the
            overall object bounding box defined by @total_size. Else, the corner of this bounding box is considered the
            origin.

        joints (None or list): Joints to use for this composite object. If None, no joints will be used
            for this top-level object. If "default", a single free joint will be added to this object.
            Otherwise, should be a list of dictionaries, where each dictionary should specify the specific
            joint attributes necessary. See http://www.mujoco.org/book/XMLreference.html#joint for reference.

        sites (None or list): list of sites to add to this composite object. If None, only the default
             object site will be used. Otherwise, should be a list of dictionaries, where each dictionary
            should specify the appropriate attributes for the given site.
            See http://www.mujoco.org/book/XMLreference.html#site for reference.

        obj_types (str or list of str): either single obj_type for all geoms or geom-specific type. Choices are
            {"collision", "visual", "all"}
    """

    def __init__(
        self,
        name,
        total_size,
        geom_types,
        geom_sizes,
        geom_locations,
        geom_quats=None,
        geom_names=None,
        geom_rgbas=None,
        geom_materials=None,
        geom_frictions=None,
        geom_condims=None,
        rgba=None,
        density=100.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        locations_relative_to_center=False,
        joints="default",
        sites=None,
        obj_types="all",
        duplicate_collision_geoms=True,
    ):
        # Always call superclass first
        super().__init__(duplicate_collision_geoms=duplicate_collision_geoms)

        self._name = name

        # Set joints
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        # Make sure all joints are named appropriately
        j_num = 0
        for joint_spec in self.joint_specs:
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(j_num)
                j_num += 1

        # Set sites
        self.site_specs = deepcopy(sites) if sites is not None else []
        # Add default site
        site_element_attr = self.get_site_attrib_template()
        site_element_attr["rgba"] = "1 0 0 0"
        site_element_attr["name"] = "default_site"
        self.site_specs.append(site_element_attr)

        # Make sure all sites are named appropriately
        s_num = 0
        for site_spec in self.site_specs:
            if "name" not in site_spec:
                site_spec["name"] = "site{}".format(s_num)
                s_num += 1

        n_geoms = len(geom_types)
        self.total_size = np.array(total_size)
        self.geom_types = np.array(geom_types)
        self.geom_sizes = deepcopy(geom_sizes)
        self.geom_locations = np.array(geom_locations)
        self.geom_quats = deepcopy(geom_quats) if geom_quats is not None else [None] * n_geoms
        self.geom_names = list(geom_names) if geom_names is not None else [None] * n_geoms
        self.geom_rgbas = list(geom_rgbas) if geom_rgbas is not None else [None] * n_geoms
        self.geom_materials = list(geom_materials) if geom_materials is not None else [None] * n_geoms
        self.geom_frictions = list(geom_frictions) if geom_frictions is not None else [None] * n_geoms
        self.geom_condims = list(geom_condims) if geom_condims is not None else [None] * n_geoms
        self.density = [density] * n_geoms if density is None or type(density) in {float, int} else list(density)
        self.solref = [solref] * n_geoms if solref is None or type(solref[0]) in {float, int} else list(solref)
        self.solimp = [solimp] * n_geoms if obj_types is None or type(solimp[0]) in {float, int} else list(solimp)
        self.rgba = rgba        # override superclass setting of this variable
        self.locations_relative_to_center = locations_relative_to_center
        self.obj_types = [obj_types] * n_geoms if obj_types is None or type(obj_types) is str else list(obj_types)

        # Always run sanity check
        self.sanity_check()

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def get_bounding_box_size(self):
        return np.array(self.total_size)

    def in_box(self, position, object_position):
        """
        Checks whether the object is contained within this CompositeObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the CompositeBoxObject as an axis-aligned grid.
        Args:
            position: 3D body position of CompositeObject
            object_position: 3D position of object to test for insertion
        """
        ub = position + self.total_size
        lb = position - self.total_size

        # fudge factor for the z-check, since after insertion the object falls to table
        lb[2] -= 0.01

        return np.all(object_position > lb) and np.all(object_position < ub)

    def _get_object_subtree(self):
        # Initialize top-level body
        obj = new_body(name="root")

        # Add all joints and sites
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        for site_spec in self.site_specs:
            obj.append(new_site(**site_spec))

        # Loop through all geoms and generate the composite object
        for i, (obj_type, g_type, g_size, g_loc, g_name, g_rgba, g_friction, g_condim,
                g_quat, g_material, g_density, g_solref, g_solimp) in enumerate(zip(
                self.obj_types,
                self.geom_types,
                self.geom_sizes,
                self.geom_locations,
                self.geom_names,
                self.geom_rgbas,
                self.geom_frictions,
                self.geom_condims,
                self.geom_quats,
                self.geom_materials,
                self.density,
                self.solref,
                self.solimp,
        )):
            # geom type
            geom_type = g_type
            # get cartesian size from size spec
            size = g_size
            cartesian_size = self._size_to_cartesian_half_lengths(geom_type, size)
            if self.locations_relative_to_center:
                # no need to convert
                pos = g_loc
            else:
                # use geom location to convert to position coordinate (the origin is the
                # center of the composite object)
                pos = [
                    (-self.total_size[0] + cartesian_size[0]) + g_loc[0],
                    (-self.total_size[1] + cartesian_size[1]) + g_loc[1],
                    (-self.total_size[2] + cartesian_size[2]) + g_loc[2],
                ]

            # geom name
            geom_name = g_name if g_name is not None else f"g{i}"

            # geom rgba
            geom_rgba = g_rgba if g_rgba is not None else self.rgba

            # geom friction
            geom_friction = array_to_string(g_friction) if g_friction is not None else \
                            array_to_string(np.array([1., 0.005, 0.0001]))  # mujoco default

            # Define base geom attributes
            geom_attr = {
                "size": size,
                "pos": pos,
                "name": geom_name,
                "type": geom_type,
            }

            # Optionally define quat if specified
            if g_quat is not None:
                geom_attr['quat'] = array_to_string(g_quat)

            # Add collision geom if necessary
            if obj_type in {"collision", "all"}:
                col_geom_attr = deepcopy(geom_attr)
                col_geom_attr.update(self.get_collision_attrib_template())
                if g_density is not None:
                    col_geom_attr['density'] = str(g_density)
                col_geom_attr['friction'] = geom_friction
                col_geom_attr['solref'] = array_to_string(g_solref)
                col_geom_attr['solimp'] = array_to_string(g_solimp)
                col_geom_attr['rgba'] = OBJECT_COLLISION_COLOR
                if g_condim is not None:
                    col_geom_attr['condim'] = str(g_condim)
                obj.append(new_geom(**col_geom_attr))

            # Add visual geom if necessary
            if obj_type in {"visual", "all"}:
                vis_geom_attr = deepcopy(geom_attr)
                vis_geom_attr.update(self.get_visual_attrib_template())
                vis_geom_attr["name"] += "_vis"
                if g_material is not None:
                    vis_geom_attr['material'] = g_material
                vis_geom_attr["rgba"] = geom_rgba
                obj.append(new_geom(**vis_geom_attr))

        return obj

    @staticmethod
    def _size_to_cartesian_half_lengths(geom_type, geom_size):
        """
        converts from geom size specification to x, y, and z half-length bounding box
        """
        if geom_type in ['box', 'ellipsoid']:
            return geom_size
        if geom_type == 'sphere':
            # size is radius
            return [geom_size[0], geom_size[0], geom_size[0]]
        if geom_type == 'capsule':
            # size is radius, half-length of cylinder part
            return [geom_size[0], geom_size[0], geom_size[0] + geom_size[1]]
        if geom_type == 'cylinder':
            # size is radius, half-length
            return [geom_size[0], geom_size[0], geom_size[1]]
        raise Exception("unsupported geom type!")

    @property
    def bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    @property
    def top_offset(self):
        return np.array([0., 0., self.total_size[2]])

    @property
    def horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)



import numpy as np

from envs.offline_objects.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict
from robosuite.utils.mjcf_utils import CustomMaterial, RED, GREEN, BLUE
import robosuite.utils.transform_utils as T


class ConeObject(CompositeObject):
    """
    Generates an approximate cone object by using cylinder or box geoms.

    Args:
        name (str): Name of this Cone object

        outer_radius (float): Radius of cone base

        inner_radius (float): Radius of cone tip (since everything is a cylinder or box)

        height (float): Height of cone

        ngeoms (int): Number of cylinder or box geoms used to approximate the cone. Use
            more geoms to make the approximation better.

        use_box (bool): If true, use box geoms instead of cylinders, corresponding to a 
            square pyramid shape instead of a conical shape.
    """

    def __init__(
        self,
        name,
        outer_radius=0.0425,
        inner_radius=0.03,
        height=0.05,
        ngeoms=8,
        use_box=False,
        rgba=None,
        material=None,
        density=1000.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        friction=None,
    ):

        # Set object attributes
        self._name = name
        self.rgba = rgba
        self.density = density
        self.friction = friction if friction is None else np.array(friction)
        self.solref = solref
        self.solimp = solimp

        self.has_material = (material is not None)
        if self.has_material:
            assert isinstance(material, CustomMaterial)
            self.material = material

        # Other private attributes
        self._important_sites = {}

        # radius of the tip and the base
        self.r1 = inner_radius
        self.r2 = outer_radius

        # number of geoms used to approximate the cone
        if ngeoms % 2 == 0:
            # use an odd number of geoms for easier computation
            ngeoms += 1
        self.n = ngeoms

        # cone height
        self.height = height

        # unit half-height for geoms
        self.unit_height = (height / ngeoms) / 2.

        # unit radius for geom radius grid
        self.unit_r = (self.r2 - self.r1) / (self.n - 1)

        self.use_box = use_box

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Optionally add material
        if self.has_material:
            self.append_material(self.material)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": [self.r2, self.r2, self.height / 2.],
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
            "solref": self.solref,
            "solimp": self.solimp,
        }
        obj_args = {}

        # stack the boxes / cylinders in the z-direction
        ngeoms_each_side = (self.n - 1) // 2
        geom_locations = [(0., 0., i * self.unit_height * 2.) for i in range(-ngeoms_each_side, ngeoms_each_side + 1)]

        if self.use_box:
            geom_sizes = [(
                self.r1 + i * self.unit_r, 
                self.r1 + i * self.unit_r,
                self.unit_height,
            ) for i in range(self.n)][::-1]
        else:
            geom_sizes = [(
                self.r1 + i * self.unit_r, 
                self.unit_height,
            ) for i in range(self.n)][::-1]

        for i in range(self.n):
            # note: set geom condim to 4 for consistency with round-nut.xml
            # geom_quat = np.array([np.cos(geom_angle / 2.), 0., 0., np.sin(geom_angle / 2.)])
            add_to_dict(
                dic=obj_args,
                geom_types="box" if self.use_box else "cylinder",
                geom_locations=geom_locations[i],
                geom_quats=None,
                geom_sizes=geom_sizes[i],
                geom_names="c_{}".format(i),
                # geom_rgbas=None if self.has_material else self.rgba,
                geom_rgbas=self.rgba,
                geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
                geom_frictions=self.friction,
                geom_condims=4,
            )

        # Sites
        obj_args["sites"] = [
            {
                "name": "center",
                "pos": (0, 0, 0),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            }
        ]

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args

class HookFrame(CompositeObject):
    """
    Generates an upside down L-shaped frame (a "hook" shape), intended to be used with StandWithMount object.

    Args:
        name (str): Name of this object

        frame_length (float): How long the frame is

        frame_height (float): How tall the frame is

        frame_thickness (float): How thick the frame is

        hook_height (float): if not None, add a box geom at the edge of the hook with this height (not half-height)

        grip_location (float): if not None, adds a grip to passed location, relative to center of the rod corresponding to @frame_height.

        grip_size ([float]): (R, H) radius and half-height for the cylindrical grip. Set to None
            to not add a grip.

        tip_size ([float]): if not None, adds a cone tip to the end of the hook for easier insertion, with the
            provided (CH, LR, UR, H) where CH is the base cylinder height, LR and UR are the lower and upper radius 
            of the cone tip, and H is the half-height of the cone tip

        friction (3-array or None): If specified, sets friction values for this object. None results in default values

        density (float): Density value to use for all geoms. Defaults to 1000

        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored

        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        frame_length=0.3,
        frame_height=0.2,
        frame_thickness=0.025,
        hook_height=None,
        grip_location=None,
        grip_size=None,
        tip_size=None,
        friction=None,
        density=1000.,
        solref=(0.02, 1.),
        solimp=(0.9, 0.95, 0.001),
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.size = None                      # Filled in automatically
        self.frame_length = frame_length
        self.frame_height = frame_height
        self.frame_thickness = frame_thickness
        self.hook_height = hook_height
        self.grip_location = grip_location
        self.grip_size = tuple(grip_size) if grip_size is not None else None
        self.tip_size = tuple(tip_size) if tip_size is not None else None
        self.friction = friction if friction is None else np.array(friction)
        self.solref = solref
        self.solimp = solimp
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.mat_name = "brass_mat"
        self.grip_mat_name = "ceramic_mat"
        self.tip_mat_name = "steel_mat"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        bin_mat = CustomMaterial(
            texture="Brass",
            tex_name="brass",
            mat_name=self.mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(bin_mat)
        # optionally add material for grip
        if (self.grip_location is not None) and (self.grip_size is not None):
            grip_mat = CustomMaterial(
                texture="Ceramic",
                tex_name="ceramic",
                mat_name=self.grip_mat_name,
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.append_material(grip_mat)
        # optionally add material for tip
        if self.tip_size is not None:
            tip_mat = CustomMaterial(
                texture="SteelScratched",
                tex_name="steel",
                mat_name=self.tip_mat_name,
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.append_material(tip_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        self.size = np.array((self.frame_length, self.frame_thickness, self.frame_height))
        if self.tip_size is not None:
            self.size[2] += 2. * (self.tip_size[0] + (2. * self.tip_size[3]))
        base_args = {
            "total_size": self.size / 2,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
            "solref": self.solref,
            "solimp": self.solimp,
        }
        obj_args = {}

        # Vertical Frame
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=((self.frame_length - self.frame_thickness) / 2, 0, -self.frame_thickness / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((self.frame_thickness, self.frame_thickness, self.frame_height - self.frame_thickness)) / 2,
            geom_names="vertical_frame",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # Horizontal Frame
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, (self.frame_height - self.frame_thickness) / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((self.frame_length, self.frame_thickness, self.frame_thickness)) / 2,
            geom_names="horizontal_frame",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # optionally add hook at the end of the horizontal frame
        if self.hook_height is not None:
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=((-self.frame_length + self.frame_thickness) / 2, 0, (self.frame_height + self.hook_height) / 2),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=np.array((self.frame_thickness, self.frame_thickness, self.hook_height)) / 2,
                geom_names="hook_frame",
                geom_rgbas=None if self.use_texture else self.rgba,
                geom_materials=self.mat_name if self.use_texture else None,
                geom_frictions=self.friction,
            )

        # optionally add a grip
        if (self.grip_location is not None) and (self.grip_size is not None):
            # note: use box grip instead of cylindrical grip for stability
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=((self.frame_length - self.frame_thickness) / 2, 0, (-self.frame_thickness / 2) + self.grip_location),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=(self.grip_size[0], self.grip_size[0], self.grip_size[1]),
                geom_names="grip_frame",
                # geom_rgbas=None if self.use_texture else self.rgba,
                geom_rgbas=(0.13, 0.13, 0.13, 1.0),
                geom_materials=self.grip_mat_name if self.use_texture else None,
                # geom_frictions=self.friction,
                geom_frictions=(1., 0.005, 0.0001), # use default friction
            )

        # optionally add cone tip
        if self.tip_size is not None:
            cone =  ConeObject(
                name="cone",
                outer_radius=self.tip_size[2],
                inner_radius=self.tip_size[1],
                height=self.tip_size[3],
                # ngeoms=8,
                ngeoms=50,
                use_box=True,
                # use_box=False,
                rgba=None,
                material=None,
                density=self.density,
                solref=self.solref,
                solimp=self.solimp,
                friction=self.friction,
            )
            cone_args = cone._get_geom_attrs()

            # DIRTY HACK: add them in reverse (in hindsight, should just turn this into a composite body...)
            cone_geom_types = cone_args["geom_types"]
            cone_geom_locations = cone_args["geom_locations"]
            cone_geom_sizes = cone_args["geom_sizes"][::-1]

            # location of mount site is the translation we need
            cylinder_offset = (
                (self.frame_length - self.frame_thickness) / 2, 
                0, 
                -self.frame_height / 2 - self.tip_size[0], # account for half-height of cylinder
            )
            cone_offset = (
                cylinder_offset[0], 
                cylinder_offset[1], 
                cylinder_offset[2] - self.tip_size[0] - self.tip_size[3] / 2., # need to move below cylinder, and account for half-height
            )

            # first add cylinder            
            add_to_dict(
                dic=obj_args,
                geom_types="cylinder",
                geom_locations=cylinder_offset,
                geom_quats=(1, 0, 0, 0),
                geom_sizes=(self.tip_size[2], self.tip_size[0]),
                geom_names="tip_cylinder",
                geom_rgbas=None if self.use_texture else self.rgba,
                geom_materials=self.tip_mat_name if self.use_texture else None,
                geom_frictions=self.friction,
            )

            # then add cone tip geoms
            for i in range(len(cone_geom_types)):
                add_to_dict(
                    dic=obj_args,
                    geom_types=cone_geom_types[i],
                    geom_locations=(
                        cone_geom_locations[i][0] + cone_offset[0],
                        cone_geom_locations[i][1] + cone_offset[1],
                        cone_geom_locations[i][2] + cone_offset[2],
                    ),
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=cone_geom_sizes[i],
                    geom_names="tip_cone_{}".format(i),
                    geom_rgbas=None if self.use_texture else self.rgba,
                    geom_materials=self.tip_mat_name if self.use_texture else None,
                    geom_frictions=self.friction,
                )

        # Sites
        obj_args["sites"] = [
            {
                "name": f"hang_site",
                "pos": (-self.frame_length / 2, 0, (self.frame_height - self.frame_thickness) / 2),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            },
            {
                "name": f"mount_site",
                "pos": ((self.frame_length - self.frame_thickness) / 2, 0, -self.frame_height / 2),
                "size": "0.002",
                "rgba": GREEN,
                "type": "sphere",
            },
            {
                "name": f"intersection_site",
                "pos": ((self.frame_length - self.frame_thickness) / 2, 0, (self.frame_height - self.frame_thickness) / 2),
                "size": "0.002",
                "rgba": BLUE,
                "type": "sphere",
            },
        ]

        if self.tip_size is not None:
            obj_args["sites"].append(
                {
                    "name": f"tip_site",
                    "pos": (
                        ((self.frame_length - self.frame_thickness) / 2),
                        0, 
                        (-self.frame_height / 2) - 2. * self.tip_size[0] - self.tip_size[3],
                    ),
                    "size": "0.002",
                    "rgba": RED,
                    "type": "sphere",
                },
            )

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args

    @property
    def init_quat(self):
        """
        Rotate the frame on its side so it is flat

        Returns:
            np.array: (x, y, z, w) quaternion orientation for this object
        """
        # Rotate 90 degrees about two consecutive axes to make the hook lie on the table instead of being upright.
        return T.quat_multiply(
                np.array([0, 0., np.sqrt(2) / 2., np.sqrt(2) / 2.]),
                np.array([-np.sqrt(2) / 2., 0., 0., np.sqrt(2) / 2.]),
            )

