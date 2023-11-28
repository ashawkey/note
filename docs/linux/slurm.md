## Slurm workload manager

Simple Linux Utility for Resource Management.



### sinfo

check information of the system.

```bash
sinfo # partitions
sinfo -N # nodes
sinfo -N --states=idle # check all idle nodes
```

to check more detailed GPU usage:

```bash
cinfo -p <partition> 
cinfo -p <partition> occupy-reserved # only show reserved quota
```



### squeue

check the current running (R) and pending (PD) jobs.

```bash
squeue -u <user name> # list your jobs
squeue -l # list all info
```



### srun/sbatch

launch/submit a job.

```bash
-J # --job-name=JOB_NAME
-p # --partition=debug, choose which queue to use
-c # --cpu-per-task=1
-n # --ntasks=N, process count (in total)
-N # --nodes=N, node count
--ntasks-per-node # process count per node (n/N)
-o # --output=OUTPUT_FILENAME
-e # --error=ERROR_FILENAME
-w # --nodelist=node[1,2], prefered nodes
-x # --exclude=node[3,5-6], nodes to avoid
--exclusive # exclusively use the nodes
--gres # --gres=gpu:2, gpu allocation
--quotatype=reserved # [phoenix-feature] auto, reserved (will not be reallocated), spot (may be reallocated)
```

QUOTA mode [phoenix-feature]:

* reserved: guaranteed GPU resources for this partition, will allocate as long as it's idle.
* spot: borrow idle resources from other partitions, will be reallocated if required by other partitions.
* auto: try to allocate reserved quota first, if not successful, turn to spot mode.



`srun` will start the job in foreground.

```bash
# quick alias
alias srcpu="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 -p 3dobject_aigc_light"
alias sr1gpu="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 -p 3dobject_aigc_light"
alias sr8gpu="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:8 -p 3dobject_aigc_light"
alias sr8gpu_spot="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:8 --quotatype=spot -p 3dobject_aigc_light"
alias squ="squeue -u `whoami`"

# use with -p!
srcpu -p <partition> bash some_script.sh
sr1gpu -p <partition> python test.py
```



`sbatch` will submit a job in background, using a script like this:

```shell
#! /bin/bash
## sbatch use \#SBATCH to state parameters!
#SBATCH -J=JOBNAME
#SBATCH --partition=gpu
#SBATCH -N=2
#SBATCH -n=32
#SBATCH --ntasks-per-node=16
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate ENV
accelerate launch main.py
```



### scontrol

re-config a pending job.

```bash
scontrol show job JOBID

scontrol hold JOBID # will not be run
scontrol update jobid=JOBID ...
scontrol release JOBID # release
```



### scancel 

delete a pending job.

```bash
scancel <jobid>
```



### sacct

show status of running and finished jobs.

```bash
sacct
```



### swatch

identify why my job is pending:

```bash
swatch check_pending <jobid>

# Reasons:
# NodeQuota: 
```

