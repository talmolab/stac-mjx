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
# # SBATCH --constraint="intel&avx2"
#SBATCH -t 1-0:00
#SBATCH -J stac-mjx
#SBATCH --gres=gpu:4
# # SBATCH -o /slurm/out
# # SBATCH -e /slurm/error
source ~/.bashrc
module load Mambaforge/22.11.1-fasrc01
source activate stac-mjx
module load cuda/12.2.0-fasrc01
nvidia-smi
python3 stac-mjx/stac_test.py -fp="fit_6iters.p" -tp="transform_6iters.p" -n=2000 --skip_fit=1 --skip_transform=0 
"""
    print(f"Submitting job")
    job_id = slurm_submit(script) 
    print(job_id)

submit()