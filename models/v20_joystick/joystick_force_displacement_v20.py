#!/usr/bin/env python3
"""
Joystick force-displacement + oscillation-frequency analysis for v20.

Adapted from janelia_model/new_joystick_model/joystick_force_displacement.py and
joystick_force_displacement_video.py, retargeted at the v20 arm+hand+joystick
model (mouse_forelimb_right_janelia_arm_hand_v20_joystick.xml) instead of the
old flat-hand reference. v20 has no position actuators (removed per Eric), so
the arm is passive; only the joystick is driven, via xfrc_applied, exactly as
the reference scripts did.

v20's joystick uses ANISOTROPIC real-joystick-measured spring/damping
coefficients (per Eric): x_slide stiffness=14.8 N/m damping=0.0099 Ns/m,
y_slide stiffness=11.2 N/m damping=0.0082 Ns/m (replacing the original
reference's shared isotropic 10 N/m / 0.0099 Ns/m guess) -- so K/C/frequency
are analyzed and reported per-axis throughout.

Three outputs:
  1. joystick_force_displacement.png -- steady-state force-vs-displacement
     sweep (8 directions x 4 magnitudes, settle 5s, record final displacement).
  2. joystick_oscillation_fft.png -- an impulse-and-ringdown test recorded at
     full timestep resolution (dt=0.001s -> 1kHz sample rate, Nyquist 500Hz,
     easily resolves a ~20Hz signal), FFT'd on x(t) and y(t) separately to find
     the actual oscillation frequency per axis.
  3. joystick_force_displacement.mp4 -- video of the impulse/ringdown test,
     rendered on the full v20 model (arm+hand+joystick visible), via OSMesa.

Required solver override (v20 is unstable under the XML's default RK4/CG at
this mass scale -- see build_v20_xml.py docstring for why):
    integrator=implicitfast, solver=Newton, iterations=50, ls_iterations=50
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = "/mnt/vast/niyo/stac-mjx/models/v20_joystick"
os.chdir(MODEL_DIR)
XML_FILE = "mouse_forelimb_right_janelia_arm_hand_v20_joystick.xml"


def load_model():
    m = mujoco.MjModel.from_xml_path(XML_FILE)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.iterations = 50
    m.opt.ls_iterations = 50
    return m


m = load_model()
d = mujoco.MjData(m)

x_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "x_slide")
y_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "y_slide")
base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "joystick_base")

Kx = m.jnt_stiffness[x_jnt]
Ky = m.jnt_stiffness[y_jnt]
Cx = m.dof_damping[x_jnt]
Cy = m.dof_damping[y_jnt]
dt = m.opt.timestep
mass = m.body_mass[base_body] + m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "joystick")]
fx_analytical = np.sqrt(Kx / mass) / (2 * np.pi)
fy_analytical = np.sqrt(Ky / mass) / (2 * np.pi)

print(f"Model: {XML_FILE}")
print(f"Integrator: implicitfast, Solver: Newton, iterations: 50")
print(f"Timestep: {dt} s")
print(f"Joint stiffness: Kx = {Kx:.2f} N/m, Ky = {Ky:.2f} N/m")
print(f"Joint damping:   Cx = {Cx:.4f} Ns/m, Cy = {Cy:.4f} Ns/m")
print(f"Joystick effective mass: {mass:.6e} kg")
print(f"Analytical undamped natural frequency: fx = {fx_analytical:.3f} Hz, fy = {fy_analytical:.3f} Hz")
print()

# ============================================================================
# Part 1: steady-state force-vs-displacement sweep (same as the reference)
# ============================================================================
force_magnitudes_mN = np.array([10, 20, 30, 40])
force_magnitudes_N = force_magnitudes_mN * 1e-3
direction_names = ["Right", "Up-Right", "Up", "Up-Left",
                   "Left", "Down-Left", "Down", "Down-Right"]
angles = np.arange(8) * (np.pi / 4)
settle_time = 5.0
settle_steps = int(settle_time / dt)

print(f"{'Direction':12s} | {'Force':>6s} | {'dx (mm)':>10s} {'dy (mm)':>10s} {'dist (mm)':>10s}")
print("-" * 65)
results_dir, results_force, results_dx, results_dy, results_dist = [], [], [], [], []

for dname, angle in zip(direction_names, angles):
    fx_hat, fy_hat = np.cos(angle), np.sin(angle)
    for f_mN, f_N in zip(force_magnitudes_mN, force_magnitudes_N):
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        for _ in range(settle_steps):
            d.xfrc_applied[base_body, 0] = f_N * fx_hat
            d.xfrc_applied[base_body, 1] = f_N * fy_hat
            mujoco.mj_step(m, d)
        dx_mm = d.qpos[m.jnt_qposadr[x_jnt]] * 1000
        dy_mm = d.qpos[m.jnt_qposadr[y_jnt]] * 1000
        dist_mm = np.sqrt(dx_mm**2 + dy_mm**2)
        results_dir.append(dname)
        results_force.append(f_mN)
        results_dx.append(dx_mm)
        results_dy.append(dy_mm)
        results_dist.append(dist_mm)
        print(f"{dname:12s} | {f_mN:4.0f} mN | {dx_mm:+10.4f} {dy_mm:+10.4f} {dist_mm:10.4f}")

results_force = np.array(results_force)
results_dist = np.array(results_dist)
results_dx = np.array(results_dx)
results_dy = np.array(results_dy)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax1 = axes[0]
colors = plt.cm.tab10(np.arange(8))
for di, dname in enumerate(direction_names):
    mask = np.array(results_dir) == dname
    ax1.plot(results_force[mask], results_dist[mask], 'o-', color=colors[di],
              label=dname, markersize=8, linewidth=2)
coeffs = np.polyfit(results_force, results_dist, 1)
fit_x = np.linspace(0, 45, 100)
ax1.plot(fit_x, np.polyval(coeffs, fit_x), 'k--', linewidth=2.5,
         label=f'Fit: {coeffs[0]:.4f} mm/mN + {coeffs[1]:.4f}')
Kavg = (Kx + Ky) / 2
ax1.plot(fit_x, fit_x / Kavg, 'r:', linewidth=2.5, label=f'Theory (avg K): {1/Kavg:.4f} mm/mN')
ax1.set_xlabel('Force (mN)'); ax1.set_ylabel('Displacement (mm)')
ax1.set_title('Force vs Displacement (v20, anisotropic Kx/Ky)')
ax1.legend(fontsize=8, loc='upper left'); ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 45); ax1.set_ylim(0, None)

ax2 = fig.add_subplot(132, projection='polar')
axes[1].remove()
for fi, f_mN in enumerate(force_magnitudes_mN):
    dir_angles, dir_dists = [], []
    for di in range(8):
        idx = di * len(force_magnitudes_mN) + fi
        dir_angles.append(angles[di]); dir_dists.append(results_dist[idx])
    dir_angles.append(dir_angles[0]); dir_dists.append(dir_dists[0])
    ax2.plot(dir_angles, dir_dists, 'o-', label=f'{f_mN:.0f} mN', markersize=6)
ax2.set_title('Displacement by Direction', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)

ax3 = axes[2]
for fi, f_mN in enumerate(force_magnitudes_mN):
    dxs, dys = [], []
    for di in range(8):
        idx = di * len(force_magnitudes_mN) + fi
        dxs.append(results_dx[idx]); dys.append(results_dy[idx])
    dxs.append(dxs[0]); dys.append(dys[0])
    ax3.plot(dxs, dys, 'o-', label=f'{f_mN:.0f} mN', markersize=8)
ax3.set_xlabel('X displacement (mm)'); ax3.set_ylabel('Y displacement (mm)')
ax3.set_title('Displacement XY Endpoints (note: elliptical, Kx != Ky)')
ax3.legend(fontsize=10)
ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='k', linewidth=0.5); ax3.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('joystick_force_displacement.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved: joystick_force_displacement.png")

# ============================================================================
# Part 2: impulse + ringdown, full-timestep-resolution recording, FFT
# ============================================================================
print("\n" + "=" * 70)
print("Impulse/ringdown oscillation-frequency test")
print("=" * 70)

impulse_force_N = 0.03       # 30 mN pulse
impulse_duration_s = 0.01    # 10 ms pulse
record_duration_s = 2.0      # record the free ringdown afterwards
impulse_steps = int(impulse_duration_s / dt)
record_steps = int(record_duration_s / dt)

mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)

n_total = impulse_steps + record_steps
t_arr = np.arange(n_total) * dt
x_arr = np.zeros(n_total)
y_arr = np.zeros(n_total)

for step in range(n_total):
    if step < impulse_steps:
        d.xfrc_applied[base_body, 0] = impulse_force_N
        d.xfrc_applied[base_body, 1] = impulse_force_N  # 45-degree pulse, drives both axes
    else:
        d.xfrc_applied[base_body, 0] = 0.0
        d.xfrc_applied[base_body, 1] = 0.0
    mujoco.mj_step(m, d)
    x_arr[step] = d.qpos[m.jnt_qposadr[x_jnt]] * 1000  # mm
    y_arr[step] = d.qpos[m.jnt_qposadr[y_jnt]] * 1000  # mm

# FFT on the post-impulse ringdown only (remove the mean/DC offset first)
ring_t = t_arr[impulse_steps:] - t_arr[impulse_steps]
ring_x = x_arr[impulse_steps:] - x_arr[impulse_steps:].mean()
ring_y = y_arr[impulse_steps:] - y_arr[impulse_steps:].mean()

freqs = np.fft.rfftfreq(len(ring_x), d=dt)
fft_x = np.abs(np.fft.rfft(ring_x))
fft_y = np.abs(np.fft.rfft(ring_y))

peak_fx = freqs[np.argmax(fft_x[1:]) + 1]  # skip DC bin
peak_fy = freqs[np.argmax(fft_y[1:]) + 1]

print(f"Impulse: {impulse_force_N*1e3:.0f} mN for {impulse_duration_s*1e3:.0f} ms, "
      f"then {record_duration_s:.1f}s free ringdown")
print(f"Sample rate: {1/dt:.0f} Hz (Nyquist {0.5/dt:.0f} Hz)")
print(f"Analytical natural frequency: fx = {fx_analytical:.3f} Hz, fy = {fy_analytical:.3f} Hz")
print(f"Measured peak frequency (x): {peak_fx:.3f} Hz")
print(f"Measured peak frequency (y): {peak_fy:.3f} Hz")
print(f"Expected (per Eric): 20 Hz")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
axes2[0, 0].plot(t_arr, x_arr, 'b-', linewidth=1)
axes2[0, 0].axvline(impulse_duration_s, color='gray', linestyle='--', label='impulse ends')
axes2[0, 0].set_xlabel('Time (s)'); axes2[0, 0].set_ylabel('x displacement (mm)')
axes2[0, 0].set_title(f'X(t): impulse + ringdown (Kx={Kx:.1f} N/m)'); axes2[0, 0].legend(); axes2[0, 0].grid(alpha=0.3)

axes2[0, 1].plot(t_arr, y_arr, 'g-', linewidth=1)
axes2[0, 1].axvline(impulse_duration_s, color='gray', linestyle='--', label='impulse ends')
axes2[0, 1].set_xlabel('Time (s)'); axes2[0, 1].set_ylabel('y displacement (mm)')
axes2[0, 1].set_title(f'Y(t): impulse + ringdown (Ky={Ky:.1f} N/m)'); axes2[0, 1].legend(); axes2[0, 1].grid(alpha=0.3)

axes2[1, 0].plot(freqs, fft_x, 'b-')
axes2[1, 0].axvline(peak_fx, color='r', linestyle='--', label=f'peak={peak_fx:.2f} Hz')
axes2[1, 0].axvline(20, color='k', linestyle=':', label='expected 20 Hz')
axes2[1, 0].axvline(fx_analytical, color='orange', linestyle=':', label=f'analytical={fx_analytical:.2f} Hz')
axes2[1, 0].set_xlim(0, 60); axes2[1, 0].set_xlabel('Frequency (Hz)'); axes2[1, 0].set_ylabel('|FFT(x)|')
axes2[1, 0].set_title('X oscillation spectrum'); axes2[1, 0].legend(); axes2[1, 0].grid(alpha=0.3)

axes2[1, 1].plot(freqs, fft_y, 'g-')
axes2[1, 1].axvline(peak_fy, color='r', linestyle='--', label=f'peak={peak_fy:.2f} Hz')
axes2[1, 1].axvline(20, color='k', linestyle=':', label='expected 20 Hz')
axes2[1, 1].axvline(fy_analytical, color='orange', linestyle=':', label=f'analytical={fy_analytical:.2f} Hz')
axes2[1, 1].set_xlim(0, 60); axes2[1, 1].set_xlabel('Frequency (Hz)'); axes2[1, 1].set_ylabel('|FFT(y)|')
axes2[1, 1].set_title('Y oscillation spectrum'); axes2[1, 1].legend(); axes2[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('joystick_oscillation_fft.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"\nPlot saved: joystick_oscillation_fft.png")

np.savez('joystick_oscillation_timeseries.npz', t=t_arr, x_mm=x_arr, y_mm=y_arr,
         freqs=freqs, fft_x=fft_x, fft_y=fft_y,
         Kx=Kx, Ky=Ky, Cx=Cx, Cy=Cy, mass=mass,
         fx_analytical=fx_analytical, fy_analytical=fy_analytical,
         peak_fx=peak_fx, peak_fy=peak_fy)
print("Raw time series + FFT saved: joystick_oscillation_timeseries.npz")
