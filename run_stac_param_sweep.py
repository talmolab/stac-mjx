#!/usr/bin/env python
import os
import sys
import logging
import time
import h5py
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Import STAC modules
import stac_mjx
from stac_mjx.stac_core import ProjectedGradient, q_loss, projection_box

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Parameter sets to try (tol gets smaller, maxiter gets larger)
PARAM_SETS = [
    {"tol": 1e-20, "maxiter": 350},
    {"tol": 1e-20, "maxiter": 600},  # baseline
    {"tol": 1e-30, "maxiter": 1000},
    {"tol": 1e-40, "maxiter": 2000},
    {"tol": 1e-50, "maxiter": 5000},
    {"tol": 1e-100, "maxiter": 8000},
    {"tol": 1e-200, "maxiter": 10000},
]


def update_solver_params(tol, maxiter):
    """
    Update the q_solver parameters in stac_core module.
    """
    logging.info(f"Updating solver params: tol={tol}, maxiter={maxiter}")
    # This modifies the solver directly in the module
    stac_mjx.stac_core.q_solver = ProjectedGradient(
        fun=q_loss, projection=projection_box, maxiter=maxiter, tol=tol
    )


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Enable XLA flags if on GPU
    stac_mjx.enable_xla_flags()

    base_path = Path("/mnt/vast/niyo/stac-mjx")
    logging.info(f"Base path: {base_path}")

    # Create output directory
    output_dir = base_path / "solver_param_sweep"
    output_dir.mkdir(exist_ok=True)
    logging.info(f"Output files will be saved to: {output_dir}")

    # Load data
    kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)
    logging.info(f"Loaded mocap data with shape {kp_data.shape}")

    # Create results log file
    results_log = output_dir / "solver_param_results.csv"
    with open(results_log, "w") as f:
        f.write("tol,maxiter,runtime_seconds,mean_error,std_error\n")

    # Run STAC with different solver parameters
    for param_set in PARAM_SETS:
        tol = param_set["tol"]
        maxiter = param_set["maxiter"]

        # Update the solver parameters
        update_solver_params(tol, maxiter)

        # Update output paths
        fit_name = f"fit_tol_{tol:.0e}_maxiter_{maxiter}.h5"
        ik_name = f"ik_tol_{tol:.0e}_maxiter_{maxiter}.h5"
        cfg.stac.fit_offsets_path = str(output_dir / fit_name)
        cfg.stac.ik_only_path = str(output_dir / ik_name)

        # Run STAC and time it
        logging.info(f"Running STAC with tol={tol}, maxiter={maxiter}")
        start_time = time.time()

        try:
            fit_path, ik_only_path = stac_mjx.run_stac(
                cfg, kp_data, sorted_kp_names, base_path=base_path
            )
            runtime = time.time() - start_time

            # Calculate error metrics from fit data
            with h5py.File(fit_path, "r") as f:
                marker_sites = np.array(f["marker_sites"])
                kp_data_saved = np.array(f["kp_data"])

                # Calculate error between marker sites and keypoint data
                error = np.sqrt(np.sum((marker_sites - kp_data_saved) ** 2, axis=1))
                mean_error = np.mean(error)
                std_error = np.std(error)

            # Log results
            with open(results_log, "a") as f:
                f.write(f"{tol},{maxiter},{runtime},{mean_error},{std_error}\n")

            logging.info(f"Completed run with tol={tol}, maxiter={maxiter}")
            logging.info(
                f"Runtime: {runtime:.2f}s, Mean error: {mean_error:.6f}, Std error: {std_error:.6f}"
            )

            # Generate visualization for this parameter set
            viz_path = output_dir / f"viz_tol_{tol:.0e}_maxiter_{maxiter}.mp4"
            stac_mjx.viz_stac(
                data_path=ik_only_path,
                n_frames=min(100, kp_data.shape[0]),
                save_path=str(viz_path),
                height=720,
                width=1280,
            )

        except Exception as e:
            logging.error(f"Error running with tol={tol}, maxiter={maxiter}: {e}")
            # Log failed run
            with open(results_log, "a") as f:
                f.write(f"{tol},{maxiter},{time.time() - start_time},FAILED,FAILED\n")

    # Generate comparison plot of errors
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        results_df = pd.read_csv(results_log)
        results_df = results_df[results_df["mean_error"] != "FAILED"]

        if not results_df.empty:
            results_df["mean_error"] = results_df["mean_error"].astype(float)
            results_df["param_label"] = results_df.apply(
                lambda row: f"tol={row['tol']:.0e}\nmaxiter={row['maxiter']}", axis=1
            )

            plt.figure(figsize=(12, 6))
            plt.bar(results_df["param_label"], results_df["mean_error"])
            plt.ylabel("Mean Error")
            plt.xlabel("Parameter Set")
            plt.title("Mean Error by Solver Parameters")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "error_comparison.png")
    except Exception as e:
        logging.error(f"Error generating comparison plot: {e}")

    logging.info("Parameter sweep completed")


if __name__ == "__main__":
    main()
