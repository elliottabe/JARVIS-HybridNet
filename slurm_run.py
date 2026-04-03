import argparse
import subprocess
import sys

def slurm_submit(script):
    """
    Submit the SLURM script using sbatch and return the job ID.
    """
    try:
        # Use a list for the command and pass the script via stdin
        output = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)

def submit(
        conda_env_name,
        script_name,
        num_gpus,
        partition,
        job_name,
        mem,
        cpus,
        time,
        note,
        dataset,
        training,
        paths,
        load_jobid,
        resume_jobid,
        override
        ):
    """
    Construct and submit the SLURM script with the specified parameters.
    """
    # GPU partition -> nodelist mappings (from sinfo -o "%P %N")
    # gpu-a40:  g[3040-3047,3050-3057,3060-3067,3070-3077] - 32 nodes
    # gpu-a100: g[3080-3087] - 8 nodes
    # gpu-l40:  g[3090-3099,3115-3119] - 15 nodes
    # gpu-l40s: g[3100-3114,3120-3124,3133-3137] - 25 nodes
    # gpu-h200: g[3125-3132] - 8 nodes
    # ckpt-g2:  g[3090-3137] - 48 GPU nodes (idle/preemptible, includes l40/l40s/h200)
    gpu_configs = {
        'gpu-a40': 'g[3040-3047,3050-3057,3060-3067,3070-3077]',
        'gpu-a100': 'g[3080-3087]',
        'gpu-l40': 'g[3090-3099,3115-3119]',
        'gpu-l40s': 'g[3100-3114,3120-3124,3133-3137]',
        'gpu-h200': 'g[3125-3132]',
        'ckpt-g2': 'g[3090-3137]',
    }

    # Auto-select nodelist based on partition (no need to specify gpu_type separately)
    gpu_resource = gpu_configs.get(partition, '')
    nodelist_line = f"#SBATCH --nodelist={gpu_resource}" if gpu_resource else ""
    
    """Submit job to cluster."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}    
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={num_gpus}
#SBATCH --mem={mem}G
#SBATCH --verbose  
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
{nodelist_line}
#SBATCH --exclude=g[3107,3115,3109]
module load cuda/12.9.1
module load gcc/12
set -x
source ~/.bashrc
nvidia-smi
conda activate {conda_env_name}
unset LD_LIBRARY_PATH
echo $SLURMD_NODENAME
python -u ./{script_name}.py paths={paths} note={note} training={training} dataset={dataset} load_jobid={load_jobid} run_id={resume_jobid if resume_jobid else '$SLURM_JOB_ID'} {override}
            """
    print(f"Submitting job")
    print(script)
    job_id = slurm_submit(script)
    print(job_id)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Submit a SLURM job with specified GPU type.')
    parser.add_argument('--conda_env_name', type=str, default='fly_neuromech',
                        help='Name of the conda environment (default: fly_neuromech)')
    parser.add_argument('--script_name', type=str, default='train_basic_imitation',
                        help='Name of the script to run (default: train_basic_imitation)')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to request (default: 8)')
    parser.add_argument('--job_name', type=str, default='Fruitfly',
                        help='Name of the SLURM job (default: Fruitfly)')
    parser.add_argument('--mem', type=int, default=128,
                        help='Memory in GB (default: 128)')
    parser.add_argument('--cpus', type=int, default=32,
                        help='Number of CPU cores (default: 32)')
    parser.add_argument('--time', type=str, default='3-00:00:00',
                        help='Time limit for the job day-hr-min-sec (default: 3-00:00:00)')
    parser.add_argument('--partition', type=str, default='gpu-l40s',
                        help='Partition to run job (default: gpu-l40s)')
    parser.add_argument('--note', type=str, default='hyak_ckpt',
                        help='Note for job (default: hyak_ckpt)')
    parser.add_argument('--dataset', type=str, default='imitation_static',
                        help='Name of dataset yaml  (default: imitation_static)')
    parser.add_argument('--training', type=str, default='ppo_basic_imitation',
                        help='Name of training yaml  (default: ppo_basic_imitation)')
    parser.add_argument('--paths', type=str, default='hyak',
                        help='Name of paths yaml  (default: hyak)')
    parser.add_argument('--load_jobid', type=str, default='',
                        help='JobID to load policy/rollout from (default: \'\')')
    parser.add_argument('--resume_jobid', type=str, default='',
                        help='JobID of a previous run to resume checkpoints from (calibrate_vnc_offline, train_closed_loop_cmaes) (default: \'\')')
    parser.add_argument('--override', type=str, default='',
                        help='Override parameters for the job (default: \'\')')

    args = parser.parse_args()

    submit(
        conda_env_name=args.conda_env_name,
        script_name=args.script_name,
        num_gpus=args.num_gpus,
        job_name=args.job_name,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        partition=args.partition,
        note=args.note,
        dataset=args.dataset,
        training=args.training,
        paths=args.paths,
        load_jobid=args.load_jobid,
        resume_jobid=args.resume_jobid,
        override=args.override,
    )

if __name__ == "__main__":
    main()
    
'''
##### Saving commands #####
#### cancel all jobs: 
squeue -u $USER -h | awk '{print $1}' | xargs scancel
 

python scripts/slurm-run_ckpt_all.py --dataset=fly_multiclip --note='hyak_ckpt'
python scripts/slurm-run_ckpt_all.py --partition=gpu-l40s --dataset=fly_multiclip --num_gpus=8 --note='hyak_multiclip'

python scripts/slurm-run_ckpt_all.py --train=train_fly_run --dataset=fly_run --note='hyak_ckpt'
python scripts/slurm-run_ckpt_all.py --partition=gpu-l40s --dataset=fly_multiclip --note='hyak_ckpt' 

## exclude nodes g3090,g3107,g3097,g3109,g3113,g3091,g3096

#SBATCH --exclude=g[3001-3007,3010-3017,3020-3027,3030-3037,3092],z[3001,3002,3005,3006]

#### full gpu node list: g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087,3090-3097,3091-3132]
#### a40 & a100 nodes only: g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087]
#### l40 & l40s nodes only: g[3091-3124]
#### h200 nodes only: g[3125-3132]

#### wandb regex: ^(?!.*table)(?!.*std).*$|^reward*&


python ./scripts/train_closed_loop.py paths=hyak dataset=imitation_static training=ppo_sensory vnc_sim=Static vnc_sim.use_sensory_action=True 
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=4 --script_name=train_closed_loop --training=ppo_sensory --dataset=imitation_static --override='vnc_sim=Static vnc_sim.use_sensory_action=True'
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_closed_loop --training=ppo_sensory --dataset=imitation_static --override='vnc_sim=Static vnc_sim.use_sensory_action=True'

#### Sensory action space training:
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_closed_loop --training=ppo_imitation  --dataset=imitation_static_gentler --override='training/network=intention_sensory vnc_sim=Static vnc_sim.use_sensory_action=True'
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_closed_loop --training=ppo_sensory  --dataset=imitation_static_gentler --override='training/network=intention_sensory vnc_sim=Static vnc_sim.use_sensory_action=True'
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_closed_loop --training=ppo_sensory  --dataset=imitation_static_gentler --override='vnc_sim=Static vnc_sim.use_sensory_action=True vnc_sim.sensory_action_per_neuron=True'

#### Motor action space training:
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=4 --script_name=train_basic_imitation --training=ppo_basic_imitation --dataset=imitation_static --override='training/network=intention_motor'
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_basic_imitation --training=ppo_basic_static --dataset=imitation_static_gentler --override='training/network=intention_motor'
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_basic_imitation --training=ppo_basic_static --dataset=imitation_static_gentler --override='training/network=intention_motor muscles.global_defaults.dyntype='muscle''
python ./scripts/slurm_run.py --partition=gpu-l40s --num_gpus=8 --script_name=train_basic_imitation --training=ppo_basic_imitation --dataset=imitation_walk_gentler --override='training/network=intention_motor'
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_basic_imitation --training=ppo_basic_imitation --dataset=imitation_walk_gentler --override='training/network=intention_motor'

python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=8 --script_name=train_basic_imitation --training=ppo_basic_static --dataset=imitation_static_gentler --override='training/network=intention_motor_ws muscles.global_defaults.dyntype='muscle''


##### Basic training
python ./scripts/slurm_run.py --partition=gpu-l40s --num_gpus=4 --script_name=train_basic_imitation --training=ppo_basic_imitation --dataset=imitation_walk_v1 --override='training/network=intention training/anatomy=v1' --time=01-12:00:00
python ./scripts/slurm_run.py --partition=gpu-l40s --num_gpus=4 --script_name=train_basic_imitation --training=ppo_basic_imitation --dataset=imitation_flight_kin --override='training/network=intention training/anatomy=v1_flight physics=flight' --time=01-12:00:00

####  CMAES Training:
python scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=1 --script_name=train_curriculum --training=cmaes_curriculum --dataset=imitation_static_linear 
python scripts/slurm_run.py --partition=gpu-l40s --num_gpus=4 --script_name=train_closed_loop_cmaes --training=cmaes_stage2 --dataset=imitation_static_robust --load_jobid=33587561 --job_name=cmaes_stage2 --override='fitness_args.use_pmap=true'
python scripts/slurm_run.py --partition=gpu-l40s --num_gpus=4 --script_name=train_closed_loop_cmaes --training=cmaes_stage3 --dataset=imitation_walk_linear --load_jobid=<STAGE2_SLURM_JOB_ID> --job_name=cmaes_stage3 --override='fitness_args.use_pmap=true'
python scripts/slurm_run.py --partition=gpu-l40s  --num_gpus=2 --script_name=train_closed_loop_cmaes --training=cmaes_stage2 --dataset=imitation_static_recovery --override='vnc_sim.bilateral_symmetry=True  cmaes_args.depressor_bias=5.0 fitness_args.chunk_length=100' --time=12:00:00 --load_jobid=70370


#### Offline calibration:
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=1 --script_name=calibrate_vnc_offline --training=cmaes_offline --dataset=imitation_static_gentler --load_jobid=34224025 --override='muscles.global_defaults.dyntype='muscle' offline_args.adam_lr_mn=0.1 offline_args.adam_lr_sn=0.1'
python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=4 --script_name=calibrate_vnc_offline --training=cmaes_offline --dataset=imitation_static_gentler --load_jobid=34224025

python ./scripts/slurm_run.py --partition=ckpt-g2 --num_gpus=1 --script_name=calibrate_vnc_offline --training=cmaes_offline --override='offline_args.optimizer=adam offline_args.rollout_path=/gscratch/portia/eabe/fly_neuromech/walk/33337384/policy_data/rollout.h5' 

python ./scripts/slurm_run.py --partition=gpu-l40s --num_gpus=2 --script_name=calibrate_vnc_offline --training=cmaes_offline --dataset=imitation_static_gentler --load_jobid=33656783 --override='vnc_sim.fix_mn_gains=True'
python ./scripts/slurm_run.py --partition=gpu-l40s --num_gpus=2 --script_name=calibrate_vnc_offline --training=cmaes_offline --dataset=imitation_static_gentler --load_jobid=33656783 --override='vnc_sim.fix_mn_gains=True cmaes_args.init_mean=100'

#### Sensory inversion:
python ./scripts/slurm_run.py --partition=gpu-l40s --num_gpus=4 --script_name=invert_sn_inputs --training=sn_inversion --load_jobid=33585197
'''