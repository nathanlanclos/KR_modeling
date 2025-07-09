#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import json
import pandas as pd
from pathlib import Path

def parse_confidence_json(json_path):
    """
    Reads a confidence JSON file and flattens its metrics into a dictionary.
    For nested dictionaries (chains_ptm, pair_chains_iptm) the keys are flattened with underscores.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics = {}
    # Top-level scalar keys
    top_keys = ['confidence_score', 'ptm', 'iptm', 'ligand_iptm', 'protein_iptm',
                'complex_plddt', 'complex_iplddt', 'complex_pde', 'complex_ipde']
    for key in top_keys:
        if key in data:
            metrics[key] = data[key]
    
    # Flatten chains_ptm
    if 'chains_ptm' in data and isinstance(data['chains_ptm'], dict):
        for chain, val in data['chains_ptm'].items():
            metrics[f'chains_ptm_{chain}'] = val

    # Flatten pair_chains_iptm
    if 'pair_chains_iptm' in data and isinstance(data['pair_chains_iptm'], dict):
        for chain_i, subdict in data['pair_chains_iptm'].items():
            if isinstance(subdict, dict):
                for chain_j, val in subdict.items():
                    metrics[f'pair_chains_iptm_{chain_i}_{chain_j}'] = val
    return metrics

def parse_affinity_json(json_path):
    """
    Reads an affinity JSON file and returns its contents as a dictionary.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except (IOError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read or parse affinity JSON {json_path}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="""
        Merge YAML CSV with Boltz replicate CSV, parse JSON confidence and affinity metrics from predictions,
        and organize final output: a summary CSV and subfolders with the final predicted files.
    """)
    parser.add_argument('--yaml_csv', required=True,
                        help="Path to the YAML-format CSV (one row per dock, with a 'yaml_file' column).")
    parser.add_argument('--boltz_csv', required=True,
                        help="Path to the Boltz CSV (one row per replicate, with columns filename, replicate, processing_time_sec).")
    parser.add_argument('--predictions_dir', required=True,
                        help="Path to the parent predictions folder that contains replicate folders (e.g. bulk_boltz_out_test).")
    parser.add_argument('--output_dir', required=True,
                        help="Path to the final output folder (will create summary CSV and subfolders).")
    parser.add_argument('--summary_csv_name', default='final_summary.csv',
                        help="Name for the final summary CSV (default: final_summary.csv)")
    args = parser.parse_args()

    # Load CSV files.
    df_yaml = pd.read_csv(args.yaml_csv)
    df_boltz = pd.read_csv(args.boltz_csv)

    # For merging, create a 'base_name' column (e.g., "ENTPD8_R249K_ATP")
    if 'yaml_file' not in df_yaml.columns:
        raise ValueError("The YAML CSV must have a 'yaml_file' column to merge on.")
    df_yaml['base_name'] = df_yaml['yaml_file'].str.replace(r'\.yaml$', '', regex=True)

    # In the Boltz CSV, extract the base_name from filenames like "ENTPD8_R249K_ATP_rep1.yaml"
    def extract_base(filename):
        if pd.isna(filename): return None
        m = re.match(r'^(.*)_rep\d+\.yaml$', filename)
        return m.group(1) if m else filename.rsplit('.', 1)[0]
    df_boltz['base_name'] = df_boltz['filename'].apply(extract_base)

    # Merge the Boltz CSV with the YAML CSV on the base dock name.
    df_merged = pd.merge(df_boltz, df_yaml, how='left', on='base_name')

    # Prepare a list to accumulate final rows.
    final_rows = []

    # For each replicate row, build the correct predictions folder path.
    for idx, row in df_merged.iterrows():
        # Derive the job_id directly from the boltz filename (e.g., "affinity_test_rep1")
        if 'filename' not in row or pd.isna(row['filename']):
            print(f"Warning: Skipping row {idx} due to missing 'filename' in Boltz CSV.")
            continue
        job_id = row['filename'].replace('.yaml', '')

        pred_folder = (Path(args.predictions_dir) /
                       f"boltz_results_{job_id}" /
                       "predictions" /
                       job_id)
        
        current_row_data = row.to_dict()

        if not pred_folder.is_dir():
            print(f"Warning: Predictions folder {pred_folder} not found for dock {job_id}. Skipping file search.")
            final_rows.append(current_row_data)
            continue

        # Create a dictionary to hold JSON metrics from all models.
        json_metrics_all = {}
        json_pattern = f"confidence_{job_id}_model_*.json"
        for json_file in pred_folder.glob(json_pattern):
            m = re.search(r'_model_(\d+)\.json$', json_file.name)
            model_idx = m.group(1) if m else "?"
            metrics = parse_confidence_json(json_file)
            for key, value in metrics.items():
                json_metrics_all[f"{key}_model_{model_idx}"] = value

        # Parse the affinity JSON file for this job.
        affinity_metrics = {}
        # Corrected pattern: affinity_<job_id>.json
        affinity_json_path = pred_folder / f"affinity_{job_id}.json"
        if affinity_json_path.is_file():
            affinity_metrics = parse_affinity_json(affinity_json_path)
        
        # Combine the original row with any found metrics
        current_row_data.update(json_metrics_all)
        current_row_data.update(affinity_metrics)
        final_rows.append(current_row_data)

    # Create final DataFrame.
    df_final = pd.DataFrame(final_rows)

    # Create output subdirectories.
    output_dir = Path(args.output_dir)
    final_pdb_dir = output_dir / "final_pdbs"
    final_pae_dir = output_dir / "final_pae"
    final_plddt_dir = output_dir / "final_plddt"
    final_conf_dir = output_dir / "final_confidence_json"
    final_affinity_json_dir = output_dir / "final_affinity_json"
    final_pre_affinity_dir = output_dir / "final_pre_affinity"
    final_pde_dir = output_dir / "final_pde"

    for subdir in [final_pdb_dir, final_pae_dir, final_plddt_dir, final_conf_dir,
                   final_affinity_json_dir, final_pre_affinity_dir, final_pde_dir]:
        subdir.mkdir(parents=True, exist_ok=True)

    # Copy predicted files for each unique job.
    unique_job_yamls = df_merged['filename'].dropna().unique()
    for job_yaml in unique_job_yamls:
        job_id = job_yaml.replace('.yaml', '')
        pred_folder = (Path(args.predictions_dir) /
                       f"boltz_results_{job_id}" /
                       "predictions" /
                       job_id)
        if not pred_folder.is_dir():
            continue

        # --- Copy all file types using the reliable job_id and corrected patterns ---
        for f in pred_folder.glob(f"{job_id}_model_*.pdb"): shutil.copy2(f, final_pdb_dir / f.name)
        for f in pred_folder.glob(f"pae_{job_id}_model_*.npz"): shutil.copy2(f, final_pae_dir / f.name)
        for f in pred_folder.glob(f"plddt_{job_id}_model_*.npz"): shutil.copy2(f, final_plddt_dir / f.name)
        for f in pred_folder.glob(f"confidence_{job_id}_model_*.json"): shutil.copy2(f, final_conf_dir / f.name)
        
        # Corrected patterns for new files
        for f in pred_folder.glob(f"affinity_{job_id}.json"): shutil.copy2(f, final_affinity_json_dir / f.name)
        for f in pred_folder.glob(f"pre_affinity_{job_id}.npz"): shutil.copy2(f, final_pre_affinity_dir / f.name)
        for f in pred_folder.glob(f"pde_{job_id}_model_*.npz"): shutil.copy2(f, final_pde_dir / f.name)

    # Write out the final summary CSV.
    summary_csv_path = output_dir / args.summary_csv_name
    df_final.to_csv(summary_csv_path, index=False)
    print(f"✅ Final summary CSV written to: {summary_csv_path}")

if __name__ == "__main__":
    main()