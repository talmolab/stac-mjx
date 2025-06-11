"""CLI script for running rodent skeletal registration"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import stac_mjx
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_and_run_stac(cfg):
    kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg)

    fit_path, ik_only_path = stac_mjx.run_stac(cfg, kp_data, sorted_kp_names)

    logging.info(
        f"Run complete. \n fit path: {fit_path} \n ik_only path: {ik_only_path}"
    )


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig):
    logging.info(f"cfg: {OmegaConf.to_yaml(cfg)}")

    stac_mjx.enable_xla_flags()

    load_and_run_stac(cfg)


if __name__ == "__main__":
    hydra_entry()
