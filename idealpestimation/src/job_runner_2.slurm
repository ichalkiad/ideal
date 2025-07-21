#!/bin/bash
#SBATCH --job-name=expII_icmd
#SBATCH --output=%x_%A_%a.out # create output files indexed with the job array task id
#SBATCH --partition=cpu_p1  #prepost
#SBATCH --time=20:00:00   # DMLE, ICM-D: 2h, CA: 30min, ICM-P: 18h (max)
#SBATCH --qos=qos_cpu-t3   #qos_cpu-dev
#SBATCH --hint=nomultithread 
#SBATCH --nodes=1          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  ############################  2 for CA, 3 for ICM-P on cpu_p1, 1 on prepost
#SBATCH --array=44-229,231-649  # expI ca, icm-p  # 0-1649 expI icm-d, mle   
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ioannis.chalkiadakis@cnrs.fr

cd ${SLURM_SUBMIT_DIR}

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
module purge
module load singularity

#index in the csv file of each parameter vector
# slurm_array_task_id=int(os.environ["SLURM_ARRAY_TASK_ID"])

set -x


# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/icm_annealing_posteriorpower_slurm.py"
time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/icm_annealing_data_slurm.py"
# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/mle_slurm.py"
# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/ca_slurm.py"

# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/aggregate_expI.py"
                                                                                                           
