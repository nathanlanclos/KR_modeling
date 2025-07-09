#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Boltz Wrapper: Runs write_yamls.py, run_boltz.py, and aggregate_job.py in sequence."
    )
    # Common / Write-YAML options
    parser.add_argument("--input_csv", required=True,
                        help="Input CSV file for write_yamls.py (protein and ligand info).")
    parser.add_argument("--output_dir", required=True,
                        help="Base output directory for all outputs. This folder will contain subfolders for YAML outputs, Boltz raw outputs, and final organized outputs.")

    # Run Boltz options (optional, passed to run_boltz.py)
    parser.add_argument("--max_time", type=int, default=None,
                        help="Max time (in minutes) for a single Boltz job (passed to run_boltz.py).")
    parser.add_argument("--num_replicates", type=int, default=None,
                        help="Number of replicates to run (passed to run_boltz.py).")
    # Allow extra options to be passed to run_boltz.py (using nargs=argparse.REMAINDER)
    parser.add_argument("--run_boltz_extra", nargs=argparse.REMAINDER,
                        help="Extra arguments to pass to run_boltz.py.")

    # Aggregate job options (optional, passed to aggregate_job.py)
    parser.add_argument("--summary_csv_name", default="final_summary.csv",
                        help="Name for the final summary CSV (passed to aggregate_job.py).")
    parser.add_argument("--aggregate_extra", nargs=argparse.REMAINDER,
                        help="Extra arguments to pass to aggregate_job.py.")

    # Write_yamls extra options
    parser.add_argument("--write_yamls_extra", nargs=argparse.REMAINDER,
                        help="Extra arguments to pass to write_yamls.py.")

    args = parser.parse_args()

    # Define common output subdirectories.
    base_out_dir = Path(args.output_dir).resolve()
    # YAML outputs (folder and CSV)
    yaml_out_dir = base_out_dir / "yaml_out"
    yaml_csv_out = base_out_dir / "yaml_output.csv"
    # Boltz raw outputs (the folder that run_boltz.py will use to write predictions and CSV)
    boltz_raw_out_dir = base_out_dir / "boltz_raw_out"
    # Final organized outputs (aggregate_job.py will create subfolders here and write the final summary CSV)
    final_out_dir = base_out_dir / "final_out"

    # Ensure base output directory exists.
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run write_yamls.py
    print("Running write_yamls.py ...")
    cmd_write = [
        "python", "write_yamls.py",
        "--input_csv", args.input_csv,
        "--yaml_out_dir", str(yaml_out_dir),
        "--csv_out", str(yaml_csv_out)
    ]
    if args.write_yamls_extra:
        cmd_write.extend(args.write_yamls_extra)
    print("Command:", " ".join(cmd_write))
    subprocess.run(cmd_write, check=True)

    # 2. Run run_boltz.py
    # run_boltz.py expects a positional argument "input_dir" containing YAML config files.
    # We pass the YAML out folder as the input.
    print("Running run_boltz.py ...")
    cmd_boltz = ["python", "run_boltz.py"]
    if args.max_time is not None:
        cmd_boltz.extend(["--max_time", str(args.max_time)])
    if args.num_replicates is not None:
        cmd_boltz.extend(["--num_replicates", str(args.num_replicates)])
    if args.run_boltz_extra:
        cmd_boltz.extend(args.run_boltz_extra)
    # Specify the Boltz output directory using --out_dir and then the input_dir (YAML configs).
    cmd_boltz.extend(["--out_dir", str(boltz_raw_out_dir), str(yaml_out_dir)])
    print("Command:", " ".join(cmd_boltz))
    subprocess.run(cmd_boltz, check=True)

    # 3. Run aggregate_job.py
    # Update: We now expect the Boltz CSV record to be named "boltz_dock_records.csv"
    # and written into the boltz_raw_out_dir.
    boltz_csv_path = boltz_raw_out_dir / "boltz_dock_records.csv"
    print("Running aggregate_job.py ...")
    cmd_agg = [
        "python", "aggregate_job.py",
        "--yaml_csv", str(yaml_csv_out),
        "--boltz_csv", str(boltz_csv_path),
        "--predictions_dir", str(boltz_raw_out_dir),
        "--output_dir", str(final_out_dir),
        "--summary_csv_name", args.summary_csv_name
    ]
    if args.aggregate_extra:
        cmd_agg.extend(args.aggregate_extra)
    print("Command:", " ".join(cmd_agg))
    subprocess.run(cmd_agg, check=True)

    print("All steps completed successfully!")
    print(f"YAML outputs are in: {yaml_out_dir}")
    print(f"YAML CSV: {yaml_csv_out}")
    print(f"Boltz raw outputs and CSV are in: {boltz_raw_out_dir}")
    print(f"Final organized outputs (including summary CSV) are in: {final_out_dir}")

if __name__ == "__main__":
    main()
