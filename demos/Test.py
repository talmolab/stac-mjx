import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
import stac_mjx 
from pathlib import Path
from jax import numpy as jp
import mediapy as media
import pandas as pd 
import hydra
from omegaconf import DictConfig


# Choose parent directory as base path to make relative pathing easier 
base_path = Path('/home/eabe/Research/MyRepos/stac-mjx') #Path.cwd().parent
stac_config_path = base_path / "configs/stac.yaml"
model_config_path = base_path / "configs/flybody.yaml"
data_dir = Path('/data/users/eabe/biomech_model/Flybody/datasets/Tuthill_data/')


stac_cfg, model_cfg = stac_mjx.load_configs(stac_config_path, model_config_path)

# tredmill_data = pd.read_csv(data_dir / "wt_berlin_linear_treadmill_dataset.csv")
# kp_names = ['head', 'thorax', 'abdomen', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3']
# coords = ['_x', '_y', '_z']
# df_names = [kp+coord for kp in kp_names for coord in coords]
# kp_data_all = tredmill_data[df_names].values
# sorted_kp_names = kp_names
# kp_data = model_cfg['MOCAP_SCALE_FACTOR']*kp_data_all.copy()

import stac_mjx.io_dict_to_hdf5 as ioh5
data_dir = Path('/data/users/eabe/biomech_model/Flybody/datasets/Tuthill_data/')
bout_dict = ioh5.load(data_dir/'bout_dict.h5')
legs_data = ['L1', 'R1', 'L2','R2', 'L3','R3']
joints_data = ['A','B','C','D','E']
sorted_kp_names = [leg + joint for leg in legs_data for joint in joints_data]
xpos_all = []
for nbout, key in enumerate(bout_dict.keys()):
    xpos_all.append(bout_dict[key]['inv_xpos'].reshape(bout_dict[key]['inv_xpos'].shape[0],-1))
kp_data = jp.concatenate(xpos_all, axis=0)
kp_data = kp_data * model_cfg['MOCAP_SCALE_FACTOR']

fit_path, transform_path = stac_mjx.run_stac(
    stac_cfg, 
    model_cfg, 
    kp_data, 
    sorted_kp_names, 
    base_path=base_path,
    save_path=data_dir
)

# set args
data_path = data_dir / "transform_tethered.p"
n_frames = 601
save_path = base_path / "videos/direct_render.mp4"

# Call mujoco_viz
frames = stac_mjx.viz_stac(data_path, stac_cfg, model_cfg, n_frames, save_path, start_frame=0, camera=1, base_path=Path.cwd().parent)

# Show the video in the notebook (it is also saved to the save_path)
media.show_video(frames, fps=model_cfg["RENDER_FPS"]) 

print('Done!')