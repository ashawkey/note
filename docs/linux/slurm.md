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
-p # --partition=xxx, choose which partition of clusters to use
-n # --ntasks=N, usually 1
-c # --cpus-per-task=16, how many CPUs to use in total (per node)
-N # --nodes=N, node count, 1 for single-node, or more for multi-node
--ntasks-per-node # must be n/N

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



`srun` will start the job in foreground, suitable for single-node training:

```bash
# quick alias (xxx is your partition)
alias srcpu="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 -p xxx"
alias sr1gpu="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 -p xxx"
alias sr8gpu="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:8 -p xxx"
alias sr8gpu_spot="srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:8 --quotatype=spot -p xxx"
alias squ="squeue -u `whoami`"

# use at ease
srcpu bash some_script.sh
sr1gpu python test.py
```



`sbatch` will submit jobs in background, and can perform multi-node training.

For example, we launch 4 * 8 = 32 GPUs to train:

```shell
#!/bin/bash
#SBATCH --job-name=MY_JOB
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --partition=3dobject_aigc_light
#SBATCH --quotatype=spot
#SBATCH --output=logs/%j_%x_out.log
#SBATCH --err=logs/%j_%x_err.log
#SBATCH --nodelist=xxx-[123-125],xxx-145

# configs
LOG_PATH="log.txt" # where all the printing goes
GPUS_PER_NODE=8 # align with --gres

echo "START TIME: $(date)"

# NCCL & AWS settings
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# proxy settings
unset http_proxy
unset https_proxy

# ip & port & rank
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=10231 # use 5 digits ports

NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable
# the accelerate config can be the same as single-node training, we will override machine rank.
export LAUNCHER="accelerate launch \
    --config_file acc_configs/gpu8.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "

export PROGRAM="\
main.py vae \
    --workspace workspce_resume \
    --resume workspace/model.safetensors
"

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
```



### scontrol

re-config a pending job.

```bash
scontrol show job JOBID

scontrol hold JOBID # will not enter running state
scontrol update jobid=JOBID ...
scontrol release JOBID # release
```



### scancel 

stop/cancel a job.

```bash
scancel <jobid>
```



### sacct

show status of running and (recently) finished jobs.

```bash
sacct

# example output
      JobID    JobName PhxPriority UserPriority VirtualPartition  Partition    Account  AllocGPUS  AllocCPUS      State ExitCode 
------------ ---------- ----------- ------------ ---------------- ---------- ---------- ---------- ---------- ---------- -------- 
10739714     accelerate        none         none xxx    llm2tmp   research          8         64 CANCELLED+      0:0 
10740830         python      normal         none xxx    llm2tmp   research          1         16    RUNNING      0:0 
10747428     accelerate        none         none xxx    llm2tmp   research          8         40  COMPLETED      0:0 
```



### swatch

identify why my job is pending.

```bash
swatch check_pending <jobid>
```

