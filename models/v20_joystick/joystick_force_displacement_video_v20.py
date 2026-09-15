#!/usr/bin/env python3
"""
Joystick impulse/ringdown video for v20 (full arm+hand+joystick model).

Renders the same impulse-then-ringdown trial used for the FFT analysis
(joystick_force_displacement_v20.py), on the full v20 model so the arm+hand
are visible together with the joystick, via OSMesa (software rendering).
v20's joystick uses anisotropic real-joystick-measured stiffness/damping
(Kx=14.8 N/m, Ky=11.2 N/m, Cx=0.0099 Ns/m, Cy=0.0082 Ns/m).

Output: joystick_force_displacement.mp4
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

MODEL_DIR = "/mnt/vast/niyo/stac-mjx/models/v20_joystick"
os.chdir(MODEL_DIR)
XML_FILE = "mouse_forelimb_right_janelia_arm_hand_v20_joystick.xml"
OUT_FILE = "joystick_force_displacement.mp4"

model = mujoco.MjModel.from_xml_path(XML_FILE)
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.iterations = 50
model.opt.ls_iterations = 50
data = mujoco.MjData(model)

x_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "x_slide")
y_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "y_slide")
base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "joystick_base")

Kx = model.jnt_stiffness[x_jnt]
Ky = model.jnt_stiffness[y_jnt]
Cx = model.dof_damping[x_jnt]
Cy = model.dof_damping[y_jnt]
dt = model.opt.timestep

print(f"Model: {XML_FILE}")
print(f"Kx={Kx:.2f} N/m, Ky={Ky:.2f} N/m, Cx={Cx:.4f} Ns/m, Cy={Cy:.4f} Ns/m, dt={dt} s")

impulse_force_N = 0.03
impulse_duration_s = 0.01
record_duration_s = 2.0
impulse_steps = int(impulse_duration_s / dt)
record_steps = int(record_duration_s / dt)
n_total = impulse_steps + record_steps

VID_W, VID_H = 640, 480
VID_FPS = 30
FRAME_SKIP = max(1, int(round(1.0 / (VID_FPS * dt))))

renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.azimuth = 90.0
camera.elevation = -35.0
camera.distance = 0.09
camera.lookat = np.array([0.022, 0.013, 0.056])

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()
    font_big = font


def add_overlay(frame, text_lines, big_line=None):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 10
    if big_line:
        draw.text((10, y), big_line, fill=(255, 255, 0), font=font_big)
        y += 28
    for line in text_lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += 20
    return np.array(img)


print(f"\nRendering {VID_W}x{VID_H} @ {VID_FPS} fps, frame_skip={FRAME_SKIP} "
      f"({n_total} sim steps -> ~{n_total // FRAME_SKIP} frames)")

writer = imageio.get_writer(OUT_FILE, fps=VID_FPS, codec="libx264", quality=7, macro_block_size=None)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

for step in range(n_total):
    if step < impulse_steps:
        data.xfrc_applied[base_body, 0] = impulse_force_N
        data.xfrc_applied[base_body, 1] = impulse_force_N
        phase = "IMPULSE (30 mN, 45deg)"
    else:
        data.xfrc_applied[base_body, 0] = 0.0
        data.xfrc_applied[base_body, 1] = 0.0
        phase = "FREE RINGDOWN"

    mujoco.mj_step(model, data)

    if step % FRAME_SKIP == 0:
        dx_mm = data.qpos[model.jnt_qposadr[x_jnt]] * 1000
        dy_mm = data.qpos[model.jnt_qposadr[y_jnt]] * 1000
        t = (step + 1) * dt

        renderer.update_scene(data, camera=camera)
        frame = renderer.render().copy()
        frame = add_overlay(frame, [
            f"Phase: {phase}",
            f"Disp:  dx={dx_mm:+.3f} mm, dy={dy_mm:+.3f} mm",
            f"Time:  {t:.3f} s",
            f"Kx={Kx:.1f} N/m, Ky={Ky:.1f} N/m (real joystick coeffs)",
            f"Cx={Cx:.4f} Ns/m, Cy={Cy:.4f} Ns/m",
            "v20: arm+hand passive (no actuators), joystick spring-loaded",
        ], big_line="Joystick impulse + ringdown (v20)")
        writer.append_data(frame)

writer.close()
print(f"\nVideo saved: {OUT_FILE}")
