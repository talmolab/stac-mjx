import subprocess 

def slurm_submit(script):
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id

def submit(lr):
    """Submit job to cluster.
    """
    fit_path = f"adam_lrtune_{lr}.p"
    transform_path = f"_.p"
    n_fit_frames = 500
    script = f"""#!/bin/bash
#SBATCH -p olveczkygpu,gpu_requeue # olveczky,cox,shared,serial_requeue # olveczkygpu,gpu_requeue
#SBATCH --mem=64000 
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --constraint="a100"
#SBATCH -t 0-5:00
#SBATCH -J adamtune
#SBATCH --gres=gpu:1
# # SBATCH -o /slurm/out
# # SBATCH -e /slurm/error
source ~/.bashrc
module load Mambaforge/22.11.1-fasrc01
source activate stac-mjx
module load cuda/12.2.0-fasrc01
nvidia-smi
python3 stac-mjx/stac_test.py paths.xml="././models/rodent_stac_optimized.xml" stac.lr={lr} paths.fit_path={fit_path} paths.transform_path={transform_path} stac.n_fit_frames={n_fit_frames}
"""
    print(f"Submitting job")
    job_id = slurm_submit(script) 
    return job_id

lrs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for lr in lrs:
    id = submit(lr)
    print(f"{id}: tol={lr} ")