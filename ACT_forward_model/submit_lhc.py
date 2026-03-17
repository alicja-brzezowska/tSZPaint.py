"""
Submit 125 y-profile jobs on a Latin Hypercube sample of Battaglia parameters.
Run with: uv run python submit_lhc.py
"""
import numpy as np
import submitit
from scipy.stats import qmc

from tszpaint.conf.config_schema import YProfileConfig
from tszpaint.y_profile.y_profile_jax import run_y_profile


# Latin Hypercube Sampling for 125 samples in the parameter space
# alpha: [0.5, 1.5], beta0: [3.0, 6.0], gamma: [0.05, 0.8], log10_P0: [-2, 2]
sampler = qmc.LatinHypercube(d=4, seed=42)
samples = qmc.scale(
    sampler.random(n=125),
    l_bounds=[0.5, 3.0, 0.05, -2.0],
    u_bounds=[1.5, 6.0, 0.80, 2.0],
)
np.savetxt("lhc_samples.txt", samples, header="alpha beta0 gamma log10_P0")


executor = submitit.AutoExecutor(folder="logs/lhc/%j")
executor.update_parameters(
    slurm_partition="icelake",
    slurm_account="HADZHIYSKA-SL3-CPU",
    cpus_per_task=32,
    mem_gb=200,
    timeout_min=6 * 60,
    slurm_array_parallelism=20,
)

configs = [
    YProfileConfig(alpha=float(r[0]), beta0=float(r[1]), gamma=float(r[2]), log10_P0=float(r[3]))
    for r in samples
]

jobs = executor.map_array(run_y_profile, configs)
print(f"Submitted {len(jobs)} jobs")
for i, (job, cfg) in enumerate(zip(jobs, configs)):
    print(f"  [{i:03d}] job={job.job_id}  alpha={cfg.alpha:.3f}  beta0={cfg.beta0:.3f}  gamma={cfg.gamma:.3f}  log10_P0={cfg.log10_P0:.3f}")
