"""CLI script for running rodent skeletal registration"""

from jax.lib import xla_bridge

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from stac_mjx import main
from stac_mjx import utils
from pathlib import Path


def load_and_run_stac(cfg):
    kp_data, sorted_kp_names = utils.load_data(cfg)

    fit_path, transform_path = main.run_stac(cfg, kp_data, sorted_kp_names)

    logging.info(
        f"Run complete. \n fit path: {fit_path} \n transform path: {transform_path}"
    )


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig):
    logging.info(f"cfg: {OmegaConf.to_yaml(cfg)}")
    # Initialize configs
    # model_cfg = hydra.compose(config_name="rodent")
    # logging.info(f"cfg: {OmegaConf.to_yaml(stac_cfg)}")
    # logging.info(f"model_cfg: {OmegaConf.to_yaml(model_cfg)}")
    # model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

    # XLA flags for Nvidia GPU
    if xla_bridge.get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
        )

    load_and_run_stac(cfg)


if __name__ == "__main__":
    hydra_entry()
