# blender script

### Install blender on ubuntu

#### Install blender 

```bash
sudo snap install blender --classic
```

This will install the blender binary and all environments.


#### Install blender as python modules

```bash
pip install bpy mathutils
```

This allow you to directly run blender scripts with python.


To enable GPU, go `Edit --> Preferences --> System --> Cycles Engine --> Choose CUDA or Optix`.

logs are printed to the terminal console, use `window --> toggle system console` to open it.


### auto import many objects

assume you have a folder with lots of obj models (each in a sub-folder):

```python
import bpy
import os

context = bpy.context
dir = r"C:\Users\hawke\Downloads\bear"

obj_dirs = os.listdir(dir)

for i, name in enumerate(obj_dirs):
    obj_dir = os.path.join(dir, name)
    files = os.listdir(obj_dir)
    for file in files:
        if not file.endswith('.obj'): continue
        path = os.path.join(obj_dir, file)
        
        print(f'[INFO] {i} process {file}')
        
        bpy.ops.import_scene.obj(filepath=path, filter_glob="*.obj;*.mtl;*.png") # also load png textures
        obj = context.selected_objects[0]

        # location (10 in a row)
        h, w = i % 10 - 5, i // 10 - 5
        obj.location = (h, w, 0)
```

### locate objects in current scene

```python
import bpy

# all objects 
bpy.data.objects

# access by index
bpy.data.objects[0]

# access by name
bpy.data.objects['Light']

# modify property
bpy.data.objects['Light'].location = (0, 0, 0)
```

### Complete render script

```python
import os
import sys
import tqdm
import math
import argparse
import numpy as np

from contextlib import contextmanager

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True)
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--camera_type", type=str, default='fixed')
parser.add_argument("--engine", type=str, default='CYCLES', choices=['BLENDER_EEVEE', 'CYCLES'])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--num_images", type=int, default=16)

parser.add_argument("--radius", type=float, default=2.0)
parser.add_argument("--fovy", type=float, default=49.1)
parser.add_argument("--bound", type=float, default=0.8)

parser.add_argument("--elevation", type=float, default=0) # +z to -z: -90 to 90
parser.add_argument("--elevation_start", type=float, default=-40)
parser.add_argument("--elevation_end", type=float, default=10)

# argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args()

# start blender env 
import bpy
from mathutils import Vector, Matrix

# render parameters
bpy.context.scene.render.engine = args.engine

bpy.context.scene.render.resolution_x = args.resolution
bpy.context.scene.render.resolution_y = args.resolution
bpy.context.scene.render.resolution_percentage = 100

bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.render.image_settings.color_mode = "RGBA"

# use nodes system
bpy.context.scene.use_nodes = True
nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links
for n in nodes:
    nodes.remove(n)
render_layers = nodes.new("CompositorNodeRLayers")

# depth
# bpy.context.view_layer.use_pass_z = True
# depth_file_output = nodes.new(type="CompositorNodeOutputFile")
# depth_file_output.label = "Depth Output"
# depth_file_output.base_path = ""
# depth_file_output.file_slots[0].use_node_format = True
# depth_file_output.format.file_format = "OPEN_EXR"
# depth_file_output.format.color_depth = "16"
# links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])

# normal 
bpy.context.view_layer.use_pass_normal = True
scale_node = nodes.new(type="CompositorNodeMixRGB")
scale_node.blend_type = "MULTIPLY"
scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs["Normal"], scale_node.inputs[1])
bias_node = nodes.new(type="CompositorNodeMixRGB")
bias_node.blend_type = "ADD"
bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_node.outputs[0], bias_node.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = "Normal Output"
normal_file_output.base_path = ""
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = "PNG"
normal_file_output.format.color_mode = "RGBA"
links.new(bias_node.outputs[0], normal_file_output.inputs[0])

# albedo
bpy.context.view_layer.use_pass_diffuse_color = True
alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = "Albedo Output"
albedo_file_output.base_path = ""
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = "PNG"
albedo_file_output.format.color_mode = "RGBA"
links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])

# NOTE: shamely, blender cannot render metallic and roughness as image...

# EEVEE will use OpenGL, CYCLES will use GPU + CUDA
if bpy.context.scene.render.engine == 'CYCLES':
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 64 # 128
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.tile_size = 8192
    
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # set which GPU to use
    for i, device in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
        if i == args.gpu:
            device.use = True
            print(f'[INFO] using device {i}: {device}')
        else:
            device.use = False
        
        
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"

# set camera
cam = bpy.context.scene.objects["Camera"]
cam.data.angle = np.deg2rad(args.fovy)

# make orbit camera
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"


def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    resolution_x_in_px = bpy.context.scene.render.resolution_x
    resolution_y_in_px = bpy.context.scene.render.resolution_y
    scale = bpy.context.scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = bpy.context.scene.render.pixel_aspect_x / bpy.context.scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float32)
    return K


def reset_scene():
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(mesh):
    """Loads a glb model into the scene."""
    if mesh.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=mesh, merge_vertices=True)
    elif mesh.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=mesh)
    else:
        raise ValueError(f"Unsupported file type: {mesh}")


def get_scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def get_scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def get_scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def normalize_scene(bound=0.9):
    # bound: normalize to [-bound, bound]

    bbox_min, bbox_max = get_scene_bbox()
    scale = 2 * bound / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = get_scene_bbox()
    offset = - (bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def save_images(object_file: str) -> None:

    object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(os.path.join(args.outdir, object_uid), exist_ok=True)

    # clean scene
    reset_scene()
    # load the object
    load_object(object_file)
    # normalize objects to [-b, b]^3
    normalize_scene(bound=args.bound)

    # create orbit camera target
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty
        
    # place cameras
    if args.camera_type == 'fixed':
        azimuths = (np.arange(args.num_images)/args.num_images*np.pi*2).astype(np.float32)
        elevations = np.deg2rad(np.asarray([args.elevation] * args.num_images).astype(np.float32))
    elif args.camera_type == 'random':
        azimuths = (np.arange(args.num_images) / args.num_images * np.pi * 2).astype(np.float32)
        elevations = np.random.uniform(args.elevation_start, args.elevation_end, args.num_images)
        elevations = np.deg2rad(elevations)
    else:
        raise NotImplementedError
    
    # get camera positions in blender coordinates
    # NOTE: assume +x axis is the object front view (azimuth = 0) 
    x = args.radius * np.cos(azimuths) * np.cos(elevations)
    y = args.radius * np.sin(azimuths) * np.cos(elevations)
    z = - args.radius * np.sin(elevations)
    cam_pos = np.stack([x,y,z], axis=-1)

    cam_poses = []
    
    for i in tqdm.trange(args.num_images):
        # set camera
        cam.location = cam_pos[i]
        bpy.context.view_layer.update()

        # pose matrix (c2w)
        c2w = np.eye(4)
        t, R = cam.matrix_world.decompose()[0:2]
        c2w[:3, :3] = np.asarray(R.to_matrix()) # [3, 3]
        c2w[:3, 3] = np.asarray(t)

        # blender to opengl
        c2w_opengl = c2w.copy()
        c2w_opengl[1] *= -1
        c2w_opengl[[1, 2]] = c2w_opengl[[2, 1]]
        
        cam_poses.append(c2w_opengl)

        # render image
        render_file_path = os.path.join(args.outdir, object_uid, f"{i:03d}")
        bpy.context.scene.render.filepath = render_file_path
        # depth_file_output.file_slots[0].path = render_file_path + "_depth"
        normal_file_output.file_slots[0].path = render_file_path + "_normal"
        albedo_file_output.file_slots[0].path = render_file_path + "_albedo"

        # if os.path.exists(render_file_path): 
        #     continue

        with stdout_redirected(): # suppress rendering logs
            bpy.ops.render.render(write_still=True)

    # write camera
    K = get_calibration_matrix_K_from_blender(cam)
    cam_poses = np.stack(cam_poses, 0)
    np.savez(os.path.join(args.outdir, object_uid, 'cameras.npz'), K=K, poses=cam_poses)

if __name__ == "__main__":
    save_images(args.mesh)
```

