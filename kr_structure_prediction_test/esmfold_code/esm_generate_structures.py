import os
import argparse
import pandas as pd
import torch
import time
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def convert_outputs_to_pdb(outputs):
    """Converts the model's output tensors to a list of PDB-formatted strings."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    final_atom_positions_np = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"].cpu().numpy()

    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i].cpu().numpy()
        pred_pos = final_atom_positions_np[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i].cpu().numpy() + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i].cpu().numpy(),
            chain_index=outputs["chain_index"][i].cpu().numpy()
            if "chain_index" in outputs
            else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def main(args):
    """Main function to run protein structure prediction and create a summary."""
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EsmForProteinFolding.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=args.low_cpu_mem_usage
    ).cuda()

    # Apply performance optimizations
    if args.use_fp16:
        model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    model.trunk.set_chunk_size(args.chunk_size)

    # 1. Prepare output directories
    structures_dir = os.path.join(args.output_directory, "structures")
    os.makedirs(structures_dir, exist_ok=True)
    print(f"✅ Output will be saved in: {os.path.abspath(args.output_directory)}")

    # 2. Process the single input CSV
    summary_data = []
    try:
        df = pd.read_csv(args.input_csv)
        if args.sequence_column not in df.columns or args.gene_column not in df.columns:
            print(f"❌ Error: Input CSV '{args.input_csv}' is missing columns '{args.sequence_column}' or '{args.gene_column}'.")
            return

        print(f"\n--- Processing {len(df)} sequences from: {args.input_csv} ---")

        # 3. Iterate over proteins, predict, and collect data
        for index, row in df.iterrows():
            sequence = str(row[args.sequence_column])
            protein_name = str(row[args.gene_column]).replace(" ", "_").replace("/", "_")

            print(f"  ({index + 1}/{len(df)}) Predicting: {protein_name}...")
            
            start_time = time.time()
            input_ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
            with torch.no_grad():
                outputs = model(input_ids)
            prediction_time = time.time() - start_time

            # Extract metrics and save PDB
            avg_plddt = outputs["plddt"][0].mean().item()
            pdbs = convert_outputs_to_pdb(outputs)
            pdb_filename = f"{protein_name}.pdb"
            pdb_path = os.path.join(structures_dir, pdb_filename)
            with open(pdb_path, "w") as f:
                f.write(pdbs[0])

            # Append data for the summary report
            summary_data.append({
                "gene_name": protein_name,
                "avg_plddt": round(avg_plddt, 2),
                "prediction_time_s": round(prediction_time, 2),
                "sequence_length": len(sequence),
                "pdb_file_path": os.path.abspath(pdb_path)
            })

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{args.input_csv}'")
        return
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        return

    # 4. Create and save the summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(args.output_directory, "summary_report.csv")
        summary_df = summary_df[['gene_name', 'avg_plddt', 'prediction_time_s', 'sequence_length', 'pdb_file_path']]
        summary_df.to_csv(summary_csv_path, index=False)
        print("\n--- 📊 Prediction Summary ---")
        print(summary_df[['gene_name', 'avg_plddt', 'prediction_time_s']])
        print(f"\n✅ Summary report saved to: {os.path.abspath(summary_csv_path)}")
    else:
        print("No structures were predicted. Summary file not created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein structures from a CSV and create a summary report.")
    
    # I/O Arguments
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_directory", type=str, required=True, help="Path to the directory where all results will be saved.")
    parser.add_argument("--sequence_column", type=str, default="sequence", help="Name of the column with protein sequences.")
    parser.add_argument("--gene_column", type=str, default="gene", help="Name of the column with protein/gene names.")

    # Model & Performance Arguments
    parser.add_argument("--model_name", type=str, default="facebook/esmfold_v1", help="Hugging Face model name.")
    parser.add_argument("--chunk_size", type=int, default=256, help="Trunk chunk size. Increase for GPUs with more VRAM (e.g., 256 for A40).")
    parser.add_argument("--use_fp16", action="store_true", default=True, help="Use float16 precision.")
    parser.add_argument("--allow_tf32", action="store_true", default=True, help="Enable TensorFloat32 math.")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="Enable low CPU memory usage for model loading.")
    
    args = parser.parse_args()
    main(args)