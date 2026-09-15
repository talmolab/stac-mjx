#!/usr/bin/env python
import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import gc
import logging
import tempfile
from pathlib import Path

# Import your STAC module (adjust the module name as needed)
import stac_mjx
from omegaconf import OmegaConf

# ------------------ Monkey-Patch OmegaConf.merge ------------------
# Override OmegaConf.merge so that it always disables strict mode on the first argument.
_original_merge = OmegaConf.merge

def custom_merge(*args, **kwargs):
    if args:
        # Disable struct mode for the first argument so extra keys are allowed
        OmegaConf.set_struct(args[0], False)
    return _original_merge(*args, **kwargs)

OmegaConf.merge = custom_merge
# ---------------------------------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------- Helper Functions ------------------------------

def get_session_info_from_csv(csv_file):
    """
    Parse the CSV filename to extract session information.
    
    Assumes CSV names like:
        reaches_<session>_tone_on.csv
    and that the session folder for 3D data is named:
        <session>_video

    Returns:
        dict: Contains session_id, data_dir, video_dir, and the registered points file.
    """
    base_csv = os.path.basename(csv_file)
    session_raw = base_csv.replace("reaches_", "").replace("_tone_on.csv", "")
    session_id = session_raw + "_video"
    subject = session_raw.split("_")[0]
    
    data_dir = f"/mnt/vast/eric/CVAT_mouse_reach/all_data/{session_id}"
    video_dir = f"/mnt/vast/eric/azim-water-reaching/dataset1/{subject}/{session_id}/videos-raw"
    registered_file = os.path.join(data_dir, f"{session_id}_mujoco_registered.h5")
    
    return {
        "session_id": session_id,
        "data_dir": data_dir,
        "video_dir": video_dir,
        "registered_file": registered_file
    }

def load_reach_events(csv_file):
    """
    Load reach events from a CSV file and compute absolute frame ranges.
    
    Returns:
        List[Tuple[int, int]]: List of (reach_start_frame, reach_end_frame) tuples.
    """
    import pandas as pd
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['reach_start_idx', 'reach_end_idx'])
    reach_events = []
    for _, row in df.iterrows():
        start_idx = int(row['start_idx'])
        reach_start_idx = int(row['reach_start_idx'])
        reach_end_idx = int(row['reach_end_idx'])
        reach_events.append((start_idx + reach_start_idx, start_idx + reach_end_idx))
    return reach_events

def extract_clip_data(registered_file, clip_range):
    """
    Extract the registered 3D points for a given clip range from the provided H5 file.
    
    Args:
        registered_file (str): Path to the _mujoco_registered.h5 file.
        clip_range (tuple): (start_frame, end_frame) indices.
    
    Returns:
        np.ndarray: Data for the clip with shape (frames, 1, nodes, 3)
    """
    with h5py.File(registered_file, 'r') as f:
        tracks = f['tracks'][:]  # Expected shape: (frames, 1, nodes, 3)
    start_frame, end_frame = clip_range
    clip_data = tracks[start_frame:end_frame+1]
    return clip_data

# ---------------------- Main STAC Pipeline ------------------------------

def main():
    # Enable XLA flags if on GPU (if needed)
    stac_mjx.enable_xla_flags()

    base_path = os.path.abspath("/mnt/vast/niyo/stac-mjx")
    logging.info(f"Base path: {base_path}")
    
    # Create output directory
    output_dir = os.path.join(base_path, "new_STACed_data")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output files will be saved to: {output_dir}")
    
    # Load both STAC and model configurations
    stac_config_dir = os.path.join(base_path, "configs/stac")
    model_config_dir = os.path.join(base_path, "configs/model")
    
    # First load the STAC config
    cfg = stac_mjx.load_configs(stac_config_dir, "stac_mouse_arm.yaml")
    logging.info("Loaded STAC config.")
    
    # Manually load the model config and merge it into the main config
    model_config = OmegaConf.load(os.path.join(model_config_dir, "mouse_arm.yaml"))
    cfg.model = model_config
    logging.info("Loaded and merged model config.")
    
    # Now cfg.model should be accessible
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.update(cfg.model, "SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "TRUNK_OPTIMIZATION_KEYPOINTS", ["Wrist"], merge=True)
    OmegaConf.update(cfg.model, "N_ITERS", 6, merge=True)
    OmegaConf.update(cfg.model, "N_SAMPLE_FRAMES", 50, merge=True)
    OmegaConf.update(cfg.model, "M_REG_COEF", 0, merge=True)
    
    # Find all CSV files that define clip ranges.
    csv_files = glob("/mnt/vast/eric/CVAT_mouse_reach/csvs/*.csv")
    logging.info(f"Found {len(csv_files)} CSV files.")

    for csv_file in csv_files:
        session_info = get_session_info_from_csv(csv_file)
        reach_events = load_reach_events(csv_file)
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        logging.info(f"Processing CSV file: {csv_file}")
        logging.info(f"Session ID: {session_info['session_id']}")
        
        # Load the registered 3D points for the session.
        reg_file = session_info["registered_file"]
        if not os.path.exists(reg_file):
            logging.error(f"Registered file not found: {reg_file}")
            continue
        with h5py.File(reg_file, 'r') as f:
            registered_tracks = f['tracks'][:]  # shape: (frames, 1, nodes, 3)
        logging.info(f"Loaded registered points from {reg_file} with shape {registered_tracks.shape}")
        
        # Process each clip (reach event) from the CSV.
        for i, clip_range in enumerate(reach_events):
            logging.info(f"Processing clip {i} with frame range {clip_range}")
            clip_data = extract_clip_data(reg_file, clip_range)  # shape: (frames, 1, nodes, 3)
            
            # Write clip_data to a temporary H5 file.
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp_filename = tmp.name
                with h5py.File(tmp_filename, "w") as tmp_h5:
                    tmp_h5.create_dataset("tracks", data=clip_data)
            logging.info(f"Temporary clip file saved to {tmp_filename}")
            
            # Load mocap data using stac_mjx.load_mocap.
            orig_data_path = cfg.stac.data_path
            cfg.stac.data_path = tmp_filename
            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, Path(base_path))
            cfg.stac.data_path = orig_data_path
            # Update n_frames_per_clip to match the clip length to avoid division errors.
            clip_length = kp_data.shape[0]
            OmegaConf.update(cfg.stac, "n_frames_per_clip", clip_length, merge=True)
            logging.info(f"Loaded mocap data for clip {i} with shape {kp_data.shape}")
            
            # Update config paths to include unique clip identifier.
            clip_suffix = f"{csv_basename}_reach{i}"
            fit_pattern = os.path.join(output_dir, f"{clip_suffix}_fit.h5")
            ik_pattern = os.path.join(output_dir, f"{clip_suffix}_ik.h5")

            # Skip if both output files already exist
            if glob(fit_pattern) and glob(ik_pattern):
                logging.info(f"Skipping clip {i}: existing outputs for {clip_suffix}")
                continue

            cfg.stac.fit_offsets_path = f"{clip_suffix}_fit.h5"
            cfg.stac.ik_only_path = f"{clip_suffix}_ik.h5"
            
            fit_path, ik_only_path = stac_mjx.run_stac(
                cfg, kp_data, sorted_kp_names, base_path=Path(output_dir)
            )
            logging.info(f"Clip {i}: saved fit to {fit_path}, ik to {ik_only_path}")

            # Use the dynamic clip length for visualization.
            n_frames = min(cfg.model.N_SAMPLE_FRAMES, kp_data.shape[0])
            viz_cfg, render = stac_mjx.viz_stac(
                data_path=ik_only_path,
                n_frames=n_frames,  # dynamically computed
                save_path=os.path.join(session_info["data_dir"], f"{csv_basename}_reach{i}_stac.mp4"),
                height=480,
                width=640,
            )
            logging.info(f"Clip {i}: STAC visualization saved.")
            
            # Remove temporary file.
            os.remove(tmp_filename)
            logging.info(f"Removed temporary clip file {tmp_filename}")
            
            gc.collect()
            
    logging.info("All CSV clips processed.")

if __name__ == "__main__":
    main()
