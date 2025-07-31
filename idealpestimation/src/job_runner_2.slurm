#!/bin/bash
#SBATCH --job-name=expI_aggrfix
#SBATCH --output=%x_%A_%a.out # create output files indexed with the job array task id
#SBATCH --partition=prepost  #cpu_p1
#SBATCH --time=05:00:00   # DMLE, ICM-D: 2h, CA: 30min, ICM-P: 18h (max)
#SBATCH --qos=qos_cpu-t3   #qos_cpu-dev
#SBATCH --hint=nomultithread 
#SBATCH --nodes=1          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2  ############################  2 for CA, 3 for ICM-P on cpu_p1, 1 on prepost
##SBATCH --array=8,11,177,187,231,250,264,272,274,281,283,289,291,299,308,310,314,327,335,352,356-357,365,389,397,404,424,436,442,444-445,468,470,643,646-647,653,666,672,681,688,690,709,716,737-738,747,749,755,757,762-784,786-931  #,1104-1164,1166-1167,1169-1218,1220-1231,1233-1247,1250-1397,1408,1570-1589,1591-1596,1598-1622,1624-1702,1704-1768,1770-1772,1774-1803,1805-1863,1866,1875,2036-2093,2095-2163,2165-2253,2255-2329,2333,2343,2502-2684,2686-2745,2747-2751,2753-2763,2765-2773,2775-2781,2783-2795,2799,2801,2807,2809,2968-3021,3023-3178,3180-3190,3192-3227,3229-3262,3266,3270,3272,3434-3458,3460-3516,3518-3704,3706-3718,3720-3727,3739,3900-4025,4027-4100,4102-4117,4119-4123,4125-4170,4172-4193,4209,4366,4368-4438,4440-4459,4461-4468,4470-4659   # expI ca, icm-p  # 0-1649 expI icm-d, mle   , expIII: ca, icmp 0-29, mle: 0-4659,
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
# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/icm_annealing_data_slurm.py"
# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/mle_slurm.py"
# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/ca_slurm.py"

time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/aggregate_expI.py"
                                                                                                           
