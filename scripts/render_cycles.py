"""Path-trace a rollout recorded by export_usd.py, with Blender Cycles.

Runs in the `blender` conda env (Blender 4.5 as a Python module), which needs
the env's own X11 libs on the library path:

    LD_LIBRARY_PATH=$HOME/miniconda3/envs/blender/lib \
        ~/miniconda3/envs/blender/bin/python scripts/render_cycles.py \
        --run renders/codesign_b2 --still 60            # one test frame
    ... --run renders/codesign_b2                       # whole clip -> video.mp4

The physics comes from the USD file untouched; this only restyles it: a large
textured ground replaces MuJoCo's floor, the robot, hazards, lane lines and
finish line get physically based materials, a sun + physical sky light the
scene, and a camera follows the robot along the corridor. Hazards glow on the
frames where something is touching them (hazard_contacts.json from the export,
the same test the cost uses).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

import bpy


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir of export_usd.py")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--still", type=int, default=None, help="render just this frame to still_<n>.png")
    ap.add_argument("--camera", choices=["behind", "chase", "side", "top"], default="behind",
                    help="behind: straight down the corridor; chase: behind and to the side")
    ap.add_argument("--ground", choices=["concrete", "soil"], default="concrete")
    ap.add_argument("--motion_blur", action="store_true", help="subtle motion blur (off: sharper stills)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- materials
HAZARD_ALPHA = {False: 0.55, True: 1.0}  # resting / touched


def principled(name, color, rough=0.5, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    return m, b


def ground_material(kind):
    """Procedural ground: no image assets, so nothing to download or license."""
    m, b = principled("ground", (0.4, 0.4, 0.4), rough=0.9)
    nt = m.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0 if kind == "concrete" else 1.5
    noise.inputs["Detail"].default_value = 12.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    if kind == "concrete":
        noise.inputs["Scale"].default_value = 0.9   # metre-scale patches
        ramp.color_ramp.elements[0].color = (0.16, 0.16, 0.155, 1)
        ramp.color_ramp.elements[1].color = (0.30, 0.295, 0.285, 1)
    else:
        ramp.color_ramp.elements[0].color = (0.13, 0.08, 0.05, 1)
        ramp.color_ramp.elements[1].color = (0.30, 0.20, 0.12, 1)
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 120.0
    fine.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25 if kind == "concrete" else 0.6
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], fine.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    color_out = ramp.outputs["Color"]
    if kind == "concrete":
        # Slab joints every 3 m: a brick texture with a thin dark mortar.
        brick = nt.nodes.new("ShaderNodeTexBrick")
        brick.offset, brick.squash = 0.0, 1.0
        brick.inputs["Scale"].default_value = 1.0 / 3.0
        brick.inputs["Mortar Size"].default_value = 0.0015
        brick.inputs["Brick Width"].default_value = 1.0
        brick.inputs["Row Height"].default_value = 1.0
        brick.inputs["Mortar"].default_value = (0.09, 0.09, 0.088, 1)
        nt.links.new(tc.outputs["Object"], brick.inputs["Vector"])
        nt.links.new(ramp.outputs["Color"], brick.inputs["Color1"])
        nt.links.new(ramp.outputs["Color"], brick.inputs["Color2"])
        color_out = brick.outputs["Color"]
    nt.links.new(color_out, b.inputs["Base Color"])
    nt.links.new(fine.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def finish_material():
    m, b = principled("finish_line", (1, 1, 1), rough=0.6)
    nt = m.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = 12.0
    chk.inputs["Color1"].default_value = (0.9, 0.9, 0.9, 1)
    chk.inputs["Color2"].default_value = (0.02, 0.02, 0.02, 1)
    nt.links.new(tc.outputs["Object"], chk.inputs["Vector"])
    nt.links.new(chk.outputs["Color"], b.inputs["Base Color"])
    return m


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


# ---------------------------------------------------------------- scene
def build(args, meta):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=meta["usd"], set_frame_range=True,
                          import_cameras=False, import_lights=False)
    sc = bpy.context.scene
    sc.render.fps = int(round(meta["fps"]))
    sc.frame_start, sc.frame_end = 0, meta["frames"] - 1

    objs = {o.name: o for o in bpy.data.objects}
    for name, o in list(objs.items()):
        if o.type == "LIGHT" or name.startswith(("Mesh_floor", "Mesh_goal")):
            bpy.data.objects.remove(o, do_unlink=True)

    # Ground: big, so the horizon is ground rather than the edge of the arena.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.scale = (80.0, 40.0, 1.0)
    # Bake the scale in, so texture coordinates are metres and the pattern is
    # not stretched across the whole plane.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign(ground, ground_material(args.ground))

    torso_mat, _ = principled("robot_body", (0.10, 0.11, 0.13), rough=0.32, metal=0.85, coat=0.3)
    leg_mat, _ = principled("robot_legs", (0.82, 0.83, 0.85), rough=0.42, coat=0.2)
    lane_mat, _ = principled("lane_paint", (0.95, 0.72, 0.06), rough=0.55)
    fin_mat = finish_material()

    hazard_names = set(meta["hazard_geoms"])
    hazards, replace_hazards = {}, []
    sc.frame_set(0)
    for o in list(bpy.data.objects):
        if o.type != "MESH" or o is ground:
            continue
        geom = o.name[len("Mesh_"):] if o.name.startswith("Mesh_") else o.name
        if geom in hazard_names:
            # The viewer's translucent blue (world.py rgba 0,0,1,0.25), lit in
            # its own hue when touched, as main.py --hazard_highlight does.
            m, b = principled(f"hz_{geom}", (0.01, 0.05, 0.60), rough=0.25, coat=0.5)
            b.inputs["Emission Color"].default_value = (0.10, 0.35, 1.0, 1.0)
            b.inputs["Emission Strength"].default_value = 0.0
            b.inputs["Alpha"].default_value = HAZARD_ALPHA[False]
            hazards[geom] = b
            replace_hazards.append((o, m))
        elif geom.startswith("corridor_line"):
            assign(o, lane_mat)
        elif geom.startswith("finish_line"):
            assign(o, fin_mat)
        elif geom.startswith("torso") or "ankle" in geom:
            # Lower legs (ankles) share the torso's finish.
            assign(o, torso_mat); smooth(o)
        else:
            assign(o, leg_mat); smooth(o)

    # Hazards never move, so swap the exporter's faceted meshes for clean,
    # slightly bevelled cylinders of the same radius, height and position.
    for o, m in replace_hazards:
        loc, dims = o.matrix_world.translation.copy(), o.dimensions.copy()
        bpy.data.objects.remove(o, do_unlink=True)
        bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=dims.x / 2, depth=dims.z,
                                            location=(loc.x, loc.y, dims.z / 2))
        cyl = bpy.context.active_object
        bev = cyl.modifiers.new("bevel", "BEVEL")
        bev.width, bev.segments = min(0.004, dims.z / 4), 3
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))
        assign(cyl, m)

    # Hazards light up on the frames something is touching them; opacity rises
    # with the glow, as in the viewer, or the translucent disc swallows it.
    for i, row in enumerate(meta["hazard_contacts"]):
        for name, hot in zip(meta["hazard_geoms"], row):
            b = hazards.get(name)
            if b is None:
                continue
            for key, value in (("Emission Strength", 2.0 if hot else 0.0),
                               ("Alpha", HAZARD_ALPHA[bool(hot)])):
                b.inputs[key].default_value = value
                b.inputs[key].keyframe_insert("default_value", frame=i)
    for mat in bpy.data.materials:
        ad = mat.node_tree.animation_data if mat.node_tree else None
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"

    # Light: physical sky without its sun disc, plus a matching sun lamp
    # (cleaner sampling than an environment-only sun).
    elev, rot = math.radians(38.0), math.radians(-35.0)
    world = bpy.data.worlds.new("sky")
    sc.world = world
    world.use_nodes = True
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = False
    sky.sun_elevation, sky.sun_rotation = elev, rot
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.25
    world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 3.2
    sun_data.angle = math.radians(1.5)
    sun = bpy.data.objects.new("sun", sun_data)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.pi / 2 - elev, 0.0, rot + math.pi / 2)

    # Camera following the torso's SMOOTHED path (1 s moving average of x and
    # y, height fixed), so it glides along the corridor and with any sideways
    # drift instead of shaking with the gait. The window is symmetric and
    # shrinks at the ends, so a steady walk is not lagged.
    torso = next(o for o in bpy.data.objects if o.name.startswith("Mesh_torso"))
    path = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        path.append(tuple(torso.matrix_world.translation[:2]))
    half, n = int(round(0.5 * meta["fps"])), len(path)
    # (camera offset from the torso, aim point this far ahead of it)
    offsets = {"behind": ((-2.2, 0.0, 1.0), 0.9), "chase": ((-1.25, -1.55, 0.75), 0.0),
               "side": ((0.0, -2.4, 0.45), 0.0), "top": ((-0.4, -0.3, 3.2), 0.0)}
    (ox, oy, oz), lead = offsets[args.camera]
    target = bpy.data.objects.new("cam_target", None)
    sc.collection.objects.link(target)
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = 40.0 if args.camera != "top" else 28.0
    for i in range(n):
        h = min(half, i, n - 1 - i)
        win = path[i - h:i + h + 1]
        x, y = (sum(p[k] for p in win) / len(win) for k in (0, 1))
        target.location = (x + lead, y, 0.12)
        cam.location = (x + ox, y + oy, oz)
        target.keyframe_insert("location", frame=sc.frame_start + i)
        cam.keyframe_insert("location", frame=sc.frame_start + i)
    print(f"camera path: x {path[0][0]:+.2f} -> {path[-1][0]:+.2f}, "
          f"y {min(p[1] for p in path):+.2f}..{max(p[1] for p in path):+.2f}")
    t = cam.constraints.new("TRACK_TO")
    t.target, t.track_axis, t.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"

    # Cycles on the GPU.
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    sc.cycles.device = "GPU"
    sc.cycles.samples = args.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 8
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    sc.render.use_motion_blur = args.motion_blur
    sc.render.motion_blur_shutter = 0.25
    w, h = (int(v) for v in args.res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.view_settings.exposure = -0.3
    sc.render.image_settings.file_format = "PNG"
    return sc


def main():
    args = parse()
    run = pathlib.Path(args.run).resolve()
    meta = json.loads((run / "hazard_contacts.json").read_text())
    sc = build(args, meta)
    if args.still is not None:
        sc.frame_set(args.still)
        sc.render.filepath = str(run / f"still_{args.still}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath)
        return
    frames = run / "frames"
    frames.mkdir(exist_ok=True)
    sc.render.filepath = str(frames / "frame_")
    bpy.ops.render.render(animation=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = run / f"video_{args.camera}.mp4"
    subprocess.run([ff, "-y", "-framerate", str(sc.render.fps), "-start_number", "0",
                    "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264", "-crf", "17",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
