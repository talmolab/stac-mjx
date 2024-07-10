import jax
from jax import numpy as jnp
from jax.lib import xla_bridge
import numpy as np

import os
import pickle
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import core.utils as utils
import core.main as main


# sets up kps for processessing and calls run_stac
def process_snips(cfg, snips_path):
    # Set up mocap data
    kp_names = utils.params["KP_NAMES"]
    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)

    # For each .p file in this directory, open it and access the kp_data attribute
    # And concatenate them together
    # shape: (250, 69)
    kp_data_list = []
    snips_order = []
    for file_name in os.listdir(snips_path):
        if file_name.endswith(".p"):
            file_path = os.path.join(snips_path, file_name)
            with open(file_path, "rb") as file:
                snips_order.append([file_path.split("/")[-1].split(".")[0]])
                snip_data = pickle.load(file)
                kp_data = snip_data["kp_data"]
                kp_data_list.append(kp_data)
    kp_data = np.vstack(kp_data_list)

    utils.params["snips_order"] = snips_order
    
    return main.run_stac(cfg, kp_data)


@hydra.main(config_path="./configs", config_name="stac", version_base=None)
def hydra_entry(cfg: DictConfig):
    # Initialize configs and convert to dictionaries
    global_cfg = hydra.compose(config_name="rodent")
    logging.info(f"cfg: {OmegaConf.to_yaml(cfg)}")
    logging.info(f"global_cfg: {OmegaConf.to_yaml(cfg)}")
    utils.init_params(OmegaConf.to_container(global_cfg, resolve=True))

    # Don't preallocate RAM?
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # XLA flags for Nvidia GPU
    if xla_bridge.get_backend().platform == "gpu":
        """
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
            "--xla_gpu_enable_async_collectives=true "
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
        )
        """
        # Set N_GPUS
        utils.params["N_GPUS"] = jax.local_device_count("gpu")

    snips_path = "./snippets_2_25_2021/snips"

    return process_snips(cfg, snips_path)


if __name__ == "__main__":

    hydra_entry()
