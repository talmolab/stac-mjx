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
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# # SBATCH -o /slurm/out
# # SBATCH -e /slurm/error
source ~/.bashrc
module load Mambaforge/22.11.1-fasrc01
source activate stac-mjx
module load cuda/12.2.0-fasrc01
nvidia-smi
python3 stac-mjx/stac_test.py -fp="LM_fit.p" -tp="LM_transform.p" -n=1000 -qt=1e-08 -s=False 
"""
    print(f"Submitting job")
    job_id = slurm_submit(script) 
    print(job_id)

submit()