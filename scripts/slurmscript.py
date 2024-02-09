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
python3 /n/home05/charleszhang/stac-mjx/stac-mjx/stac_test.py --fit_path={"no_kpstoopt_fit.p"} --transform_path={"no_kpstoopt_transform.p"} --n_fit_frames={500} --tol={1e-03}
"""
    print(f"Submitting job")
    job_id = slurm_submit(script) 
    print(job_id)

submit()