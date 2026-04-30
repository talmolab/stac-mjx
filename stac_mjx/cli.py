"""Command-line interface for running STAC-MJX."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import stac_mjx
from stac_mjx.config import compose_config


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments and return args plus Hydra override list.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.

    Returns:
        Tuple of (parsed args, Hydra overrides).
    """
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


def run_pipeline(
    cfg: DictConfig,
    base_path: Path,
    enable_xla: bool = True,
) -> tuple[str, str | None]:
    """Execute the STAC pipeline given a composed config.

    Args:
        cfg: STAC configuration.
        base_path: Base path for resolving relative paths.
        enable_xla: Whether to set XLA flags before running.

    Returns:
        Tuple of (fit_offsets path, ik_only path or None).
    """
    if enable_xla:
        stac_mjx.enable_xla_flags()

    kp_data, sorted_kp_names = stac_mjx.load_data(cfg, base_path=base_path)
    return stac_mjx.run_stac(cfg, kp_data, sorted_kp_names, base_path=base_path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.

    Returns:
        Exit code (0 on success).
    """
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
