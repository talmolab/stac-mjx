import jax
from jax import numpy as jnp
from jax.lib import xla_bridge
import numpy as np

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from stac_mjx import main
from stac_mjx import utils
from pathlib import Path


def load_and_run_stac(stac_cfg, model_cfg):
    base_path = Path.cwd()

    data_path = base_path / stac_cfg.data_path
    kp_data, sorted_kp_names = utils.load_data(data_path, model_cfg)

    fit_path, transform_path = main.run_stac(
        stac_cfg, model_cfg, kp_data, sorted_kp_names, base_path=base_path
    )

    logging.info(
        f"Run complete. \n fit path: {fit_path} \n transform path: {transform_path}"
    )


@hydra.main(config_path="./configs", config_name="stac", version_base=None)
def hydra_entry(stac_cfg: DictConfig):
    # Initialize configs
    model_cfg = hydra.compose(config_name="rodent")
    logging.info(f"cfg: {OmegaConf.to_yaml(stac_cfg)}")
    logging.info(f"model_cfg: {OmegaConf.to_yaml(model_cfg)}")
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

    # XLA flags for Nvidia GPU
    if xla_bridge.get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
        )

    load_and_run_stac(stac_cfg, model_cfg)


if __name__ == "__main__":
    hydra_entry()
