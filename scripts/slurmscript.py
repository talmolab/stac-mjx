import subprocess 
def slurm_submit(script):
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id

def submit():
    """Submit job to cluster.
    """
    script = f"""#!/bin/bash
#SBATCH -p olveczkygpu,gpu_requeue # olveczky,cox,shared,serial_requeue # olveczkygpu,gpu_requeue
#SBATCH --mem=64000 
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --constraint="a100"
#SBATCH -t 0-3:00
#SBATCH -J lbfgs
#SBATCH --gres=gpu:1
# # SBATCH -o /slurm/out
# # SBATCH -e /slurm/error
source ~/.bashrc
module load Mambaforge/22.11.1-fasrc01
source activate stac-mjx
module load cuda/12.2.0-fasrc01
nvidia-smi
python3 stac-mjx/stac_test.py paths.xml="././models/rodent_stac_optimized.xml" paths.fit_path="fit_half2_1k_12_21_1.p" paths.transform_path="transform_half2_1k_12_21_1.p" stac.sampler="random" stac.n_fit_frames=1000
"""
    print(f"Submitting job")
    job_id = slurm_submit(script) 
    print(job_id)

submit()