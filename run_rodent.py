import jax
from jax import numpy as jp
from jax.lib import xla_bridge
import numpy as np

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from stac_mjx import main
from stac_mjx import utils


@hydra.main(config_path="./configs", config_name="stac", version_base=None)
def hydra_entry(cfg: DictConfig):
    # Initialize configs and convert to dictionaries
    global_cfg = hydra.compose(config_name=cfg.paths.model_config)
    logging.info(f"cfg: {OmegaConf.to_yaml(cfg)}")
    logging.info(f"global_cfg: {OmegaConf.to_yaml(global_cfg)}")
    utils.init_params(OmegaConf.to_container(global_cfg, resolve=True))

    # XLA flags for Nvidia GPU
    if xla_bridge.get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
        )
        # Set N_GPUS
        utils.params["N_GPUS"] = jax.local_device_count("gpu")

    # Set up mocap data
    kp_names = utils.params["KP_NAMES"]
    # argsort returns the indices that sort the array to match the order of marker sites
    stac_keypoint_order = np.argsort(kp_names)
    data_path = cfg.paths.data_path

    # Load kp_data, /1000 to scale data (from mm to meters)
    kp_data = utils.loadmat(data_path)["pred"][:] / 1000

    # Preparing DANNCE data by reordering and reshaping
    # Resulting kp_data is of shape (n_frames, n_keypoints)
    kp_data = jp.array(kp_data[:, :, stac_keypoint_order])
    kp_data = jp.transpose(kp_data, (0, 2, 1))
    kp_data = jp.reshape(kp_data, (kp_data.shape[0], -1))

    return main.run_stac(cfg, kp_data)


if __name__ == "__main__":
    fit_path, transform_path = hydra_entry()
    logging.info(
        f"Run complete. \n fit path: {fit_path} \n transform path: {transform_path}"
    )
