import argparse
import os
import random
import subprocess
from pathlib import Path

import submitit
from yacs.config import CfgNode

from config.default import get_cfg_defaults
from helpers.utils import generate_run_id
from main import get_parser as get_main_parser
from main import parse_arg_changes


def parse_args_and_main_args(**parser_kwargs):
    main_parser = get_main_parser(**parser_kwargs)
    main_args, remaining_argv = main_parser.parse_known_args()
    parser = argparse.ArgumentParser()  # NOTE: since there is not parent parser, the --help will only show description for main arguments  # noqa: E501
    parser.add_argument("--no_jobid", action="store_true", help="Remove job id at the beginning of name")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=24 * 3 * 60, type=int, help="Duration of the job")
    parser.add_argument("--partition", default="your_partition", type=str, help="Partition where to submit")
    parser.add_argument("--qos", default=None, type=str, help="Job priority")
    parser.add_argument("--use_volta32", action="store_true", help="Request V100-32gb GPUs")
    parser.add_argument("--use_ampere", action="store_true", help="Request A100 GPUs")
    parser.add_argument("--comment", default="", type=str, help="Comment for job scheduler")
    parser.add_argument("--job_name", default="Flowception", type=str, help="Job name for SLURM")
    
    parser.add_argument("--mem_gb", type=int, help="CPU memory per gpu", default=0)
    parser.add_argument("--deepspeed", action="store_true", help="Use deepspeed")
    parser.add_argument("--cluster_type", default="slurm", choices=["slurm", "local", "debug"])
    parser.add_argument("--cluster", default="default", type=str, help="Cluster name (used for run ID generation)")
    parser.add_argument("--accelerate_debug", action="store_true")
    parser.add_argument(
        "--autoencoder_training",
        action="store_true",
        help="Launch autoencoder training instead of diffusion model training.",
    )
    parser.add_argument(
        "--autoencoder_watermarking",
        action="store_true",
        help="Finetuning autoencoder decoder for watermarking.",
    )

    args = parser.parse_args(remaining_argv)
    return args, main_args


def args_to_command(args: argparse.Namespace) -> str:
    command = []
    print("ARGS IN COMMAND :")
    for arg, value in vars(args).items():
        print(arg, value)
        if value:
            command.append("--" + arg)
            if isinstance(value, list):
                command.extend(map(str, value))  # handle list arguments
            elif value is not True:  # for boolean flags
                command.append(str(value))
    return " ".join(command)


class Task(submitit.helpers.Checkpointable):
    def __init__(self, args: argparse.Namespace, main_args: argparse.Namespace, main_cfg: CfgNode):
        self.args = args
        self.main_args = main_args
        self.main_cfg = main_cfg

    def __call__(self):
        print("RUN ID: ", self.main_cfg.RUN_ID)
        print("Exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment()
        job_id = dist_env._job_env.job_id
        rng = random.Random(job_id)
        dist_env.master_port = rng.randint(10000, 20000)
        dist_env.world_size = self.args.ngpus * self.args.nodes
        dist_env.local_world_size = self.args.ngpus
        dist_env = dist_env.export(set_cuda_visible_devices=False)

        exp_path = dist_env._job_env.paths.folder.parent
        self.main_args.name = exp_path.name  # this guarantees consistency with outdir in main.train

        print("JOB ID: ", job_id)
        self.main_cfg.defrost()
        self.main_cfg.JOB_ID = int(job_id)
        self.main_cfg.freeze()

        cfg_path = exp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            f.write(self.main_cfg.dump())  # save config to file
        self.main_args.config = str(cfg_path)  # use the updated config file
        self.main_args.append = []  # appended args are already stored into config

        # uncomment this block to provide extra env variables
        extra_env_vars = {

            "FI_EFA_SET_CUDA_SYNC_MEMOPS": "0",
            # "NCCL_SOCKET_IFNAME": "en",
            "FI_EFA_USE_HUGE_PAGE": "0",
            "FI_EFA_FORK_SAFE": "1",


            "MKL_THREADING_LAYER": "OMP",
        }
        os.environ.update(**extra_env_vars)

        main_options = args_to_command(self.main_args)

        # Setup accelerate config, see here for possible args:
        # https://huggingface.co/docs/accelerate/v0.13.2/en/package_reference/cli
        accelerate_options = (
            f"--multi_gpu "
            f"--mixed_precision=bf16 "
            f"--num_processes={dist_env.world_size} "
            f"--num_machines={self.args.nodes} "
            f"--machine_rank={dist_env.rank} "
            f"--main_process_ip={dist_env.master_addr} "
            f"--main_process_port={dist_env.master_port} "
            f"--dynamo_backend=no "
        )

        # Add deepspeed config if using it
        if self.args.deepspeed and self.args.nodes > 1:
            hostfile_dir = exp_path / "hostfiles"
            hostfile_dir.mkdir(parents=True, exist_ok=True)
            hostfile = (hostfile_dir / f"{job_id}.txt").resolve()
            if dist_env.rank == 0:
                with open(hostfile, "w") as f:
                    for host in dist_env._job_env.hostnames:
                        f.write(f"{host} slots={self.args.ngpus}\n")
                print(f"Created hostfile: {hostfile}")
            accelerate_options += (
                f"--use_deepspeed "
                f"--deepspeed_hostfile {hostfile} "
                f"--deepspeed_multinode_launcher standard "
            )

        if self.args.accelerate_debug:
            accelerate_options += "--debug "

        if args.autoencoder_training:
            cmd = f"accelerate launch {accelerate_options}  -m modules.autoencoders.main {main_options}"
        elif args.autoencoder_watermarking:
            cmd = f"accelerate launch {accelerate_options}  -m modules.autoencoders.main_watermark {main_options}"
        else:
            cmd = f"accelerate launch {accelerate_options} ./main.py {main_options}"
        print(f"Running command:\n{cmd}")
        self.print_env()
        if dist_env.local_rank == 0:
            subprocess.check_call(cmd.split())
        else:
            print("Waiting for master to finish")

    def checkpoint(self, *args, **kwargs):
        self.main_args.resume = True
        print("Requeuing ", "args", self.args, "main_args", self.main_args)
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

    def print_env(self):
        print("=============================================")
        env = os.environ
        dist_vars = [
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
        ]
        print("Torch distributed env:")
        print("\n".join(f"{k}={env[k]}" for k in dist_vars if k in env))
        print("=============================================")
        print("NCCL env:")
        print("\n".join(f"{k}={env[k]}" for k in env if "NCCL" in k))
        print("=============================================")
        if self.args.cluster_type == "slurm":
            print("Slurm env:")
            print(
                "\n".join(
                    f"{k}={env[k]}" for k in sorted(env.keys()) if k.startswith(("SLURM_", "SUBMITIT_"))
                )
            )
            print("=============================================")


if __name__ == "__main__":
    args, main_args = parse_args_and_main_args()
    print("args", args, "main_args", main_args)

    main_arg_changes = parse_arg_changes(main_args.append)
    main_arg_changes += ["DATA.CLUSTER", args.cluster]

    cfg = get_cfg_defaults()
    cfg.merge_from_file(main_args.config)
    cfg.merge_from_list(main_arg_changes)
    cfg["RUN_ID"] = cfg["RUN_ID"] or generate_run_id(args.cluster)
    cfg.freeze()

    main_args.name = "%A_" + main_args.name
    if args.no_jobid:
        main_args.name = main_args.name[3:]
    root_path = Path(main_args.logdir) / main_args.name

    executor = submitit.AutoExecutor(
        folder=root_path / "submitit_logs",
        cluster=args.cluster_type,
        slurm_max_num_timeout=3,
    )

    generic_kwargs = {
        "gpus_per_node": args.ngpus,
        "tasks_per_node": 1,
        "timeout_min": args.timeout,
    }

    executor.update_parameters(**generic_kwargs)

    if args.cluster_type == "slurm":
        gpu_type = args.constraint if hasattr(args, "constraint") else ""

        print(f"Submitting to partition: {args.partition} {gpu_type}")

        slurm_kwargs = {
            "slurm_job_name": args.job_name,
            "slurm_partition": args.partition,
            "slurm_nodes": args.nodes,
            "slurm_cpus_per_task": 10 * args.ngpus,
            "slurm_time": args.timeout,
            "slurm_exclude": "",
            "slurm_mem_gb": args.mem_gb * args.ngpus,
            "slurm_constraint": gpu_type,
            "slurm_gpus_per_node": args.ngpus,
            "slurm_tasks_per_node": 1,
            "slurm_signal_delay_s": 120,
        }

        if args.qos is not None:
            slurm_kwargs["slurm_qos"] = args.qos

        executor.update_parameters(**slurm_kwargs)

    task = Task(args, main_args, cfg)
    job = executor.submit(task)
    print(f"Submitted job: {job.job_id}")
