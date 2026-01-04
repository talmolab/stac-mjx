"""Command-line interface for running STAC-MJX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import stac_mjx
from stac_mjx import io


def parse_args(argv: Sequence[str] | None = None) -> Tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments and return the args plus Hydra override list."""
    parser = argparse.ArgumentParser(
        description="Run STAC-MJX inverse kinematics from the command line."
    )
    parser.add_argument(
        "--config-path",
        default="configs",
        help="Path to Hydra config directory (default: configs)",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name to load (default: config)",
    )
    parser.add_argument(
        "--base-path",
        default=str(Path.cwd()),
        help="Base path for resolving data/model paths in the config (default: CWD)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    parser.add_argument(
        "--skip-xla-flags",
        action="store_true",
        help="Do not set XLA flags before running.",
    )

    args, overrides = parser.parse_known_args(argv)
    return args, overrides


def compose_config(
    config_path: Path | str,
    config_name: str,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    """Load and validate the Hydra configuration."""
    overrides = list(overrides or [])
    overrides.extend(["hydra/job_logging=disabled", "hydra/hydra_logging=disabled"])

    config_dir = Path(config_path).resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    structured_config = OmegaConf.structured(io.Config)
    merged_cfg = OmegaConf.merge(structured_config, cfg)
    logging.debug("Config loaded and validated.")
    return merged_cfg


def run_pipeline(
    cfg: DictConfig,
    base_path: Path,
    enable_xla: bool = True,
) -> tuple[str, str | None]:
    """Execute the STAC pipeline given a composed config."""
    if enable_xla:
        stac_mjx.enable_xla_flags()

    kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path=base_path)
    return stac_mjx.run_stac(
        cfg, kp_data, sorted_kp_names, base_path=base_path
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)

    args, hydra_overrides = parse_args(argv)
    base_path = Path(args.base_path).resolve()

    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=hydra_overrides,
    )

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return 0

    fit_path, ik_only_path = run_pipeline(
        cfg=cfg,
        base_path=base_path,
        enable_xla=not args.skip_xla_flags,
    )

    logging.info("Run complete.")
    logging.info("Fit path: %s", fit_path)
    logging.info("IK-only path: %s", ik_only_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
