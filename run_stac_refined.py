#!/usr/bin/env python
# filepath: /mnt/vast/niyo/stac-mjx/run_stac_refined.py
import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import gc
import logging
import tempfile
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import traceback

os.environ['JAX_PLATFORM_NAME'] = 'gpu'

# Set up logging first
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Import STAC module
import stac_mjx
import jax
logging.info(f"Using JAX devices: {jax.devices()}")

# ---------------------- Helper Functions ------------------------------

def get_session_info(session_id):
    """
    Generate paths for the session data.
    
    Args:
        session_id (str): Session identifier like 'A36-1_2023-07-18_16-54-01_lightOff'
        
    Returns:
        dict: Contains data_dir, video_dir, and registered_file
    """
    subject = session_id.split("_")[0]
    session_video_id = f"{session_id}_video"
    
    data_dir = f"/mnt/vast/eric/CVAT_mouse_reach/all_data/{session_video_id}"
    video_dir = f"/mnt/vast/eric/azim-water-reaching/dataset1/{subject}/{session_video_id}/videos-raw"
    registered_file = os.path.join(data_dir, f"{session_video_id}_mujoco_registered.h5")
    
    return {
        "session_id": session_video_id,
        "data_dir": data_dir,
        "video_dir": video_dir,
        "registered_file": registered_file,
    }

def load_refined_reach_events(csv_path):
    """
    Load reach events from the refined CSV file.
    
    Args:
        csv_path (str): Path to the refined reaches CSV
        
    Returns:
        pd.DataFrame: DataFrame with valid reaches and their frame ranges
    """
    df = pd.read_csv(csv_path)
    # Filter for valid trials (where both start and end are not 0)
    valid_trials = df[(df['start'] != 0) & (df['end'] != 0)].copy()
    
    # Convert columns to integers to avoid slice index errors
    valid_trials['trial'] = valid_trials['trial'].astype(int)
    valid_trials['trial_start'] = valid_trials['trial_start'].astype(int)
    valid_trials['start'] = valid_trials['start'].astype(int)
    valid_trials['end'] = valid_trials['end'].astype(int)
    
    # Calculate absolute frame indices for reach start
    valid_trials['reach_start_frame'] = valid_trials['trial_start'] + valid_trials['start']
    
    return valid_trials

def extract_clip_data(registered_file, start_frame, n_frames=60):
    """
    Extract the registered 3D points for a given frame range.
    
    Args:
        registered_file (str): Path to the _mujoco_registered.h5 file
        start_frame (int): Starting frame index
        n_frames (int): Number of frames to extract (default: 100)
        
    Returns:
        np.ndarray: Data for the clip with shape (frames, 1, nodes, 3)
    """
    # Ensure integer indices
    start_frame = int(start_frame)
    end_frame = start_frame + n_frames  # This will make slicing give us exactly n_frames
    
    with h5py.File(registered_file, "r") as f:
        tracks = f["tracks"][:]  # Expected shape: (frames, 1, nodes, 3)
        total_frames = tracks.shape[0]
        
        # Adjust start_frame if needed to ensure we get exactly n_frames
        if end_frame > total_frames:
            # If we can't get n_frames starting from start_frame, 
            # move the window back to get a full n_frames
            frames_available = total_frames - start_frame
            if frames_available < n_frames:
                adjustment = n_frames - frames_available
                start_frame = max(0, start_frame - adjustment)
                end_frame = start_frame + n_frames
                logging.warning(f"Adjusted frame window to {start_frame}-{end_frame} to ensure {n_frames} frames")
            else:
                # Just clip end_frame if we have enough frames
                end_frame = total_frames
        
        # Extract clip data
        clip_data = tracks[start_frame:end_frame]
        
        # Verify we have exactly n_frames
        actual_frames = clip_data.shape[0]
        if actual_frames != n_frames:
            logging.warning(f"Failed to extract exactly {n_frames} frames. Got {actual_frames} frames.")
            
            # If we have too few frames, pad by repeating the last frame
            if actual_frames < n_frames:
                padding_needed = n_frames - actual_frames
                last_frame = clip_data[-1:].repeat(padding_needed, axis=0)
                clip_data = np.concatenate([clip_data, last_frame], axis=0)
                logging.info(f"Padded clip with {padding_needed} repeated frames to reach {n_frames} frames")
            # If we have too many frames, trim
            elif actual_frames > n_frames:
                clip_data = clip_data[:n_frames]
                logging.info(f"Trimmed clip to exactly {n_frames} frames")
    
    # Final verification
    assert clip_data.shape[0] == n_frames, f"Failed to ensure exactly {n_frames} frames. Got {clip_data.shape[0]}"
    
    return clip_data

def ensure_config_structure(cfg, base_path):
    """
    Ensure the configuration has the proper structure for STAC.
    
    Args:
        cfg (DictConfig): Original configuration
        base_path (str): Base path for configuration files
        
    Returns:
        DictConfig: Updated configuration with proper structure
    """
    # Make the configuration mutable
    OmegaConf.set_struct(cfg, False)
    
    # Create empty structures if they don't exist
    if not hasattr(cfg, 'stac'):
        cfg.stac = OmegaConf.create({})
    
    # Make sure we have a model section
    if not hasattr(cfg, 'model'):
        # Try to get the model name from the configuration
        model_name = 'mouse_arm'
        if hasattr(cfg, 'model'):
            model_name = cfg.model if isinstance(cfg.model, str) else model_name
        
        # Load the model configuration
        model_config_path = os.path.join(base_path, f"configs/model/{model_name}.yaml")
        if os.path.exists(model_config_path):
            model_config = OmegaConf.load(model_config_path)
            cfg.model = model_config
            logging.info(f"Loaded model config from {model_name}")
        else:
            logging.warning(f"Could not find model config at {model_config_path}")
    
    # Ensure stac has required fields
    if not hasattr(cfg.stac, 'data_path'):
        cfg.stac.data_path = "/mnt/vast/niyo/stac-mjx/mujoco_aligned_tracking.h5"
        
    if not hasattr(cfg.stac, 'n_frames_per_clip'):
        cfg.stac.n_frames_per_clip = 100
    
    return cfg

# ---------------------- Main STAC Pipeline ------------------------------

@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm", version_base=None)
def main(cfg: DictConfig):
    # Enable XLA flags if on GPU
    stac_mjx.enable_xla_flags()
    
    base_path = os.path.abspath("/mnt/vast/niyo/stac-mjx")
    logging.info(f"Base path: {base_path}")
    
    # Create output directory
    output_dir = os.path.join(base_path, "refined_STACed_data_janelia")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output files will be saved to: {output_dir}")
    
    # Ensure proper configuration structure
    cfg = ensure_config_structure(cfg, base_path)
    
    # Update model configuration
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.update(cfg.model, "SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "TRUNK_OPTIMIZATION_KEYPOINTS", ["Wrist"], merge=True)
    OmegaConf.update(cfg.model, "N_ITERS", 6, merge=True)
    OmegaConf.update(cfg.model, "N_SAMPLE_FRAMES", 50, merge=True)
    OmegaConf.update(cfg.model, "M_REG_COEF", 0, merge=True)
    
    # Load the refined reach events
    refined_csv_path = "/mnt/vast/niyo/stac-mjx/refined_reaches/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv"
    valid_reaches = load_refined_reach_events(refined_csv_path)
    logging.info(f"Loaded {len(valid_reaches)} valid reaches from refined CSV")
    
    # Get session information
    session_id = "A36-1_2023-07-18_16-54-01_lightOff"
    session_info = get_session_info(session_id)
    
    # Load the registered 3D points for the session
    reg_file = session_info["registered_file"]
    if not os.path.exists(reg_file):
        logging.error(f"Registered file not found: {reg_file}")
        return
    
    # Get total number of frames
    with h5py.File(reg_file, "r") as f:
        total_frames = f["tracks"].shape[0]
    logging.info(f"Loaded registered points from {reg_file} with {total_frames} total frames")
    
    # Process each valid reach
    for i, reach in valid_reaches.iterrows():
        # Ensure integer values
        trial_num = int(reach['trial'])
        start_frame = int(reach['reach_start_frame'])
        
        logging.info(f"Processing trial {trial_num} (reach {i}) with start frame {start_frame}")
        
        try:
            # Extract clip data for this reach - always exactly 100 frames
            clip_data = extract_clip_data(reg_file, start_frame, n_frames=100)
            
            # Verify clip data shape
            logging.info(f"Extracted clip with shape {clip_data.shape}")
            if clip_data.shape[0] != 100:
                logging.error(f"Clip extraction failed for trial {trial_num}: expected 100 frames, got {clip_data.shape[0]}")
                continue
            
            # Create a permanent file rather than a temporary one
            clip_dir = os.path.join(output_dir, "reach_clips")
            os.makedirs(clip_dir, exist_ok=True)
            clip_file = os.path.join(clip_dir, f"{session_id}_trial{trial_num:02d}_clip.h5")
            
            # Save the clip data
            with h5py.File(clip_file, "w") as h5f:
                h5f.create_dataset("tracks", data=clip_data)
            logging.info(f"Reach clip saved to {clip_file}")
            
            # Store original data path and ensure stac section is mutable
            OmegaConf.set_struct(cfg.stac, False)
            orig_data_path = cfg.stac.data_path
            
            # Set the clip file as the new data path
            cfg.stac.data_path = clip_file
            
            # Load mocap data
            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, Path(base_path))
            
            # Restore original data path
            cfg.stac.data_path = orig_data_path
            
            # Update n_frames_per_clip to match the clip length (always 100)
            clip_length = kp_data.shape[0]
            OmegaConf.update(cfg.stac, "n_frames_per_clip", clip_length, merge=True)
            logging.info(f"Loaded mocap data for trial {trial_num} with shape {kp_data.shape}")
            
            # Update config paths to include unique clip identifier
            clip_suffix = f"{session_id}_trial{trial_num:02d}"
            fit_pattern = os.path.join(output_dir, f"{clip_suffix}_fit.h5")
            ik_pattern = os.path.join(output_dir, f"{clip_suffix}_ik.h5")
            
            # Skip if both output files already exist
            if glob(fit_pattern) and glob(ik_pattern):
                logging.info(f"Skipping trial {trial_num}: existing outputs for {clip_suffix}")
                continue
                
            cfg.stac.fit_offsets_path = f"{clip_suffix}_fit.h5"
            cfg.stac.ik_only_path = f"{clip_suffix}_ik.h5"
            
            fit_path, ik_only_path = stac_mjx.run_stac(
                cfg, kp_data, sorted_kp_names, base_path=Path(output_dir)
            )
            logging.info(f"Trial {trial_num}: saved fit to {fit_path}, ik to {ik_only_path}")
            
            # Use the dynamic clip length for visualization
            n_frames = min(cfg.model.N_SAMPLE_FRAMES, kp_data.shape[0])
            viz_cfg, render = stac_mjx.viz_stac(
                data_path=ik_only_path,
                n_frames=n_frames,
                save_path=os.path.join(output_dir, f"{clip_suffix}_stac.mp4"),
                height=480,
                width=640,
            )
            logging.info(f"Trial {trial_num}: STAC visualization saved")
            
        except Exception as e:
            # Print full stack trace for better debugging
            logging.error(f"Error processing trial {trial_num}: {str(e)}")
            logging.error(traceback.format_exc())
        
        # Clean up memory
        gc.collect()
    
    logging.info("All valid reaches processed")

if __name__ == "__main__":
    main()