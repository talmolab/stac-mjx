import subprocess 

# submits a set of jobs with different params
def slurm_submit(script):
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id

def submit(tol):
    """Submit job to cluster.
    """
    fit_path = f"tol_test_fit2k_{tol}.p"
    transform_path = f"tol_test_transform_{tol}.p"
    n_fit_frames = 2000
    script = f"""#!/bin/bash
#SBATCH -p olveczkygpu,gpu_requeue # olveczky,cox,shared,serial_requeue # olveczkygpu,gpu_requeue
#SBATCH --mem=32000 
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
nvidia-smi
python3 /n/home05/charleszhang/stac-mjx/stac-mjx/stac_test.py --fit_path={fit_path} --transform_path={transform_path} --n_fit_frames={n_fit_frames} --tol={tol}
"""
    print(f"Submitting job")
    # Returns the job id
    return slurm_submit(script) 

# 1e-2 -> 1e-5
tols=[1.0e-2,1.0e-3, 1.0e-4, 1.0e-5]
for tol in tols:
    id = submit(tol)
    print(f"{id}: tol={tol} ")