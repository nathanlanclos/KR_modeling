#!/usr/bin/env python3
import argparse
import subprocess
import os
import shutil
import time
import csv
import re

def duplicate_yaml_files(input_dir, num_replicates):
    """
    Creates a subfolder named 'replicates' inside input_dir,
    copies all YAML files, and duplicates each file num_replicates times,
    appending '_repN' to the filename.
    Returns the path to the replicates folder.
    """
    replicates_dir = os.path.join(input_dir, "replicates")
    os.makedirs(replicates_dir, exist_ok=True)
    
    yaml_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(input_dir, f))
    ]
    
    for file in yaml_files:
        file_path = os.path.join(input_dir, file)
        base, ext = os.path.splitext(file)
        for rep in range(1, num_replicates + 1):
            new_filename = f"{base}_rep{rep}{ext}"
            new_filepath = os.path.join(replicates_dir, new_filename)
            shutil.copy(file_path, new_filepath)
    return replicates_dir

def construct_command(args, yaml_filepath):
    """
    Constructs the boltz predict command with the provided arguments and YAML file.
    In this updated version, we simply pass the user-specified output directory without appending an extra subfolder.
    """
    cmd = ["boltz", "predict"]
    
    # Use the provided out_dir directly.
    cmd += ["--out_dir", args.out_dir]
    
    # Add the rest of the options.
    if args.cache:
        cmd += ["--cache", args.cache]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.devices:
        cmd += ["--devices", str(args.devices)]
    if args.accelerator:
        cmd += ["--accelerator", args.accelerator]
    if args.recycling_steps is not None:
        cmd += ["--recycling_steps", str(args.recycling_steps)]
    if args.sampling_steps is not None:
        cmd += ["--sampling_steps", str(args.sampling_steps)]
    if args.diffusion_samples is not None:
        cmd += ["--diffusion_samples", str(args.diffusion_samples)]
    if args.step_scale is not None:
        cmd += ["--step_scale", str(args.step_scale)]
    if args.write_full_pae:
        cmd.append("--write_full_pae")
    if args.write_full_pde:
        cmd.append("--write_full_pde")
    if args.output_format:
        cmd += ["--output_format", args.output_format]
    if args.num_workers is not None:
        cmd += ["--num_workers", str(args.num_workers)]
    if args.override:
        cmd.append("--override")
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.use_msa_server:
        cmd.append("--use_msa_server")
    if args.msa_server_url:
        cmd += ["--msa_server_url", args.msa_server_url]
    if args.msa_pairing_strategy:
        cmd += ["--msa_pairing_strategy", args.msa_pairing_strategy]
    
    # Append the YAML file path (the configuration file) as the final argument.
    cmd.append(yaml_filepath)
    return cmd

def extract_replicate_number(filename):
    """
    Extract replicate number from filename if present.
    Expected format: <basename>_rep<replicate>.<ext>
    Returns 1 if no replicate number is found.
    """
    match = re.search(r"_rep(\d+)", filename)
    return int(match.group(1)) if match else 1

def main():
    parser = argparse.ArgumentParser(
        description='Run Boltz structure prediction for a folder full of YAML configuration files.'
    )
    
    # Positional argument: input directory containing YAML files.
    parser.add_argument('input_dir', help='Directory containing YAML configuration files.')
    
    # Extra arguments
    parser.add_argument('--max_time', type=float, default=None,
                        help='Maximum time (in minutes) for a single job before moving to the next file.')
    parser.add_argument('--num_replicates', type=int, default=1,
                        help='Number of times to run the same job (replicates).')
    
    # Boltz predict options
    parser.add_argument('--out_dir', type=str, required=True,
                        help='The path where to save the predictions. This is now used as the destination folder for all Boltz outputs.')
    parser.add_argument('--cache', type=str, help='Directory to download the data and model. Default is ~/.boltz.')
    parser.add_argument('--checkpoint', type=str,
                        help='Optional checkpoint. Will use the provided Boltz-1 model by default.')
    parser.add_argument('--devices', type=int, default=1,
                        help='The number of devices to use for prediction. Default is 1.')
    parser.add_argument('--accelerator', type=str, choices=['gpu','cpu','tpu'], default='gpu',
                        help='The accelerator to use for prediction. Default is gpu.')
    parser.add_argument('--recycling_steps', type=int, default=3,
                        help='The number of recycling steps to use for prediction. Default is 3.')
    parser.add_argument('--sampling_steps', type=int, default=200,
                        help='The number of sampling steps to use for prediction. Default is 200.')
    parser.add_argument('--diffusion_samples', type=int, default=1,
                        help='The number of diffusion samples to use for prediction. Default is 1.')
    parser.add_argument('--step_scale', type=float, default=1.638,
                        help='Step size (related to diffusion temperature). Default is 1.638.')
    # --write_full_pae should default to True
    parser.add_argument('--no-write_full_pae', dest='write_full_pae', action='store_false',
                        help='Do not dump the pae into a npz file. (Default writes full pae)')
    parser.set_defaults(write_full_pae=True)
    parser.add_argument('--write_full_pde', action='store_true',
                        help='Dump the pde into a npz file. Default is False.')
    parser.add_argument('--output_format', type=str, choices=['pdb','mmcif'], default='mmcif',
                        help='Output format for predictions. Default is mmcif.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers to use for prediction. Default is 2.')
    parser.add_argument('--override', action='store_true',
                        help='Override existing predictions. Default is False.')
    parser.add_argument('--seed', type=int, help='Seed for the random number generator.')
    parser.add_argument('--use_msa_server', action='store_true',
                        help='Use the MMSeqs2 server for MSA generation. Default is False.')
    parser.add_argument('--msa_server_url', type=str,
                        help='MSA server URL. Used only if --use_msa_server is set.')
    parser.add_argument('--msa_pairing_strategy', type=str, choices=['greedy', 'complete'],
                        help="Pairing strategy to use. Used only if --use_msa_server is set.")
    
    args = parser.parse_args()
    
    # Always create a replicates folder, even if num_replicates == 1.
    jobs_dir = duplicate_yaml_files(args.input_dir, args.num_replicates)
    print(f"Replicate files created in: {jobs_dir}")
    
    # List all YAML files in the jobs directory.
    job_files = [
        f for f in os.listdir(jobs_dir)
        if f.endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(jobs_dir, f))
    ]
    
    # List to hold CSV records.
    records = []
    
    # Process each YAML file.
    for job_file in job_files:
        yaml_path = os.path.join(jobs_dir, job_file)
        replicate_number = extract_replicate_number(job_file)
        
        # Build the boltz predict command.
        cmd = construct_command(args, yaml_path)
        print(f"Running command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            # Use timeout if max_time is specified (convert minutes to seconds).
            if args.max_time:
                timeout_sec = args.max_time * 60
                subprocess.run(cmd, timeout=timeout_sec, check=True)
            else:
                subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
        except subprocess.TimeoutExpired:
            elapsed = args.max_time * 60 if args.max_time else (time.time() - start_time)
            print(f"Job {job_file} exceeded max_time. Moving to next file.")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"Job {job_file} failed with error: {e}. Moving to next file.")
        
        # Record job details.
        records.append({
            'filename': job_file,
            'replicate': replicate_number,
            'processing_time_sec': round(elapsed, 2)
        })
    
    # Write the CSV record file to the out_dir.
    csv_filepath = os.path.join(args.out_dir, "boltz_dock_records.csv")
    with open(csv_filepath, mode='w', newline='') as csvfile:
        fieldnames = ['filename', 'replicate', 'processing_time_sec']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    
    print(f"CSV record file saved at {csv_filepath}")

if __name__ == "__main__":
    main()
