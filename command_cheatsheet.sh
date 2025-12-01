# show which graphic cards are available
sinfo -p gpunodes -o "%20N  %10m  %25f  %20G "

# check job status
squeue -u $USER
slurm_report

# submit a job
sbatch job.sh
srun --partition gpunodes \
     --cpus-per-task=4 \
     --mem=30G \
     --gres=gpu:rtx_a2000:1 \
     job.sh

# run interative session
srun --partition gpunodes \
     --cpus-per-task=4 \
     --mem=30G \
     --gres=gpu:rtx_4090:1 \
     --pty bash --login