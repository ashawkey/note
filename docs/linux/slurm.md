## Slurm workload manager

Simple Linux Utility for Resource Management.



### sinfo

check information of the system.

```bash
sinfo # partitions
sinfo -N # nodes
sinfo -N --states=idle # check all idle nodes
```



### squeue

check the current running (R) and pending (PD) jobs.

```bash
squeue -u <user name> # list your jobs
squeue -l # list all
squeue -j jobid
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
```

`srun` will start the job in foreground.

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

delete a job.



### saact

show finished jobs.