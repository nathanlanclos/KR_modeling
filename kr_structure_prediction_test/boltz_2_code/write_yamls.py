#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
import yaml
import json
import sys

# Custom YAML representers to ensure specific formatting for lists and strings.
# This makes the output YAML file cleaner and more readable.
class FlowStyleList(list):
    pass

def represent_flow_style_list(dumper, data):
    """YAML representer for FlowStyleList for inline, comma-separated lists."""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowStyleList, represent_flow_style_list)

class SingleQuotedString(str):
    pass

def single_quoted_representer(dumper, data):
    """YAML representer to force single quotes for strings."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

yaml.add_representer(SingleQuotedString, single_quoted_representer)

def generate_template_csv(filepath="boltz_template.csv"):
    """
    Generates a template CSV file with all possible headers and an example row.
    """
    headers = [
        'foldname',
        # --- Entities (Proteins, Ligands, etc.) ---
        # You can have as many entities as you need by adding more columns (entity_3_..., entity_4_...)
        'entity_1_type', 'entity_1_id', 'entity_1_sequence', 'entity_1_smiles', 'entity_1_ccd', 'entity_1_msa', 'entity_1_cyclic', 'entity_1_modifications',
        'entity_2_type', 'entity_2_id', 'entity_2_sequence', 'entity_2_smiles', 'entity_2_ccd', 'entity_2_msa', 'entity_2_cyclic', 'entity_2_modifications',
        # --- Constraints (JSON formatted) ---
        'bonds', 'pockets', 'contacts',
        # --- Properties ---
        'affinity_binder'
    ]
    
    example_data = {
        'foldname': ['protein_and_ligand_example'],
        # Entity 1: A protein chain
        'entity_1_type': ['protein'],
        'entity_1_id': ['A'],
        'entity_1_sequence': ['PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK'],
        'entity_1_smiles': [''],
        'entity_1_ccd': [''],
        'entity_1_msa': ['path/to/proteinA.msa'],
        'entity_1_cyclic': [False],
        'entity_1_modifications': ['[{"position": 10, "ccd": "SEP"}]'],
        # Entity 2: A ligand defined by SMILES
        'entity_2_type': ['ligand'],
        'entity_2_id': ['B'],
        'entity_2_sequence': [''],
        'entity_2_smiles': ['CC(=O)O'],
        'entity_2_ccd': [''],
        'entity_2_msa': [''],
        'entity_2_cyclic': [''],
        'entity_2_modifications': [''],
        # Constraints examples
        'bonds': ['[{"atom1": ["A", 10, "OG"], "atom2": ["B", 2, "O1"]}]'],
        'pockets': ['[{"binder": "B", "contacts": [["A", 25, "CA"], ["A", 30]], "max_distance": 12.0}]'],
        'contacts': [''],
        # Properties example
        'affinity_binder': ['B']
    }
    
    # Fill in empty values for headers not in the example
    for header in headers:
        if header not in example_data:
            example_data[header] = ['']

    df = pd.DataFrame(example_data, columns=headers)
    df.to_csv(filepath, index=False)
    print(f"Template CSV file saved to: {filepath}")
    print("Please edit this file to define your own protein-ligand complexes.")

def process_row(row):
    """
    Processes one row of the CSV to generate the YAML structure.
    """
    # Initialize the main dictionary structure for the YAML file.
    yaml_dict = {}
    
    # --- 1. Process all entities (proteins, ligands, etc.) ---
    sequences = []
    entity_pattern = re.compile(r'entity_(\d+)_type')
    entity_cols = [col for col in row.index if entity_pattern.match(col)]
    
    # Sort by entity number to maintain order
    entity_keys_sorted = sorted(entity_cols, key=lambda x: int(entity_pattern.match(x).group(1)))
    
    for type_key in entity_keys_sorted:
        entity_type = row[type_key]
        if pd.isna(entity_type) or entity_type.strip() == '':
            continue
            
        entity_index = entity_pattern.match(type_key).group(1)
        
        entity_obj = {}
        entity_data = {}

        # The ID can be a single letter or a FlowStyleList for multiple identical chains
        chain_ids = str(row[f'entity_{entity_index}_id']).split(',')
        entity_data['id'] = FlowStyleList(chain_ids) if len(chain_ids) > 1 else chain_ids[0]

        # Add sequence-like properties
        if not pd.isna(row.get(f'entity_{entity_index}_sequence')):
            entity_data['sequence'] = row[f'entity_{entity_index}_sequence']
        if not pd.isna(row.get(f'entity_{entity_index}_smiles')):
            entity_data['smiles'] = SingleQuotedString(row[f'entity_{entity_index}_smiles'])
        if not pd.isna(row.get(f'entity_{entity_index}_ccd')):
            entity_data['ccd'] = row[f'entity_{entity_index}_ccd']
        if not pd.isna(row.get(f'entity_{entity_index}_msa')):
            entity_data['msa'] = row[f'entity_{entity_index}_msa']
        if not pd.isna(row.get(f'entity_{entity_index}_cyclic')) and row[f'entity_{entity_index}_cyclic']:
             entity_data['cyclic'] = bool(row[f'entity_{entity_index}_cyclic'])

        # Handle modifications as a JSON string
        mod_col = f'entity_{entity_index}_modifications'
        if mod_col in row and not pd.isna(row[mod_col]) and row[mod_col].strip():
            try:
                entity_data['modifications'] = json.loads(row[mod_col])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in column '{mod_col}'. Skipping modifications for this entity.", file=sys.stderr)

        entity_obj[entity_type] = entity_data
        sequences.append(entity_obj)
        
    if sequences:
        yaml_dict['sequences'] = sequences

    # --- 2. Process Constraints (bonds, pockets, contacts) ---
    constraints = []
    constraint_keys = ['bonds', 'pockets', 'contacts']
    for key in constraint_keys:
        if key in row and not pd.isna(row[key]) and row[key].strip():
            try:
                # The column contains a JSON string representing a list of constraint dicts
                parsed_json = json.loads(row[key])
                if isinstance(parsed_json, list):
                    # Wrap each item in the list with its key
                    for item in parsed_json:
                        constraints.append({key.rstrip('s'): item}) # e.g., 'bonds' -> 'bond'
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in constraint column '{key}'. Skipping.", file=sys.stderr)
    
    if constraints:
        yaml_dict['constraints'] = constraints

    # --- 3. Process Properties (e.g., affinity) ---
    properties = []
    if 'affinity_binder' in row and not pd.isna(row['affinity_binder']):
        properties.append({'affinity': {'binder': row['affinity_binder']}})
    
    if properties:
        yaml_dict['properties'] = properties
        
    return yaml_dict

def main():
    """Main function to parse arguments and drive the script."""
    parser = argparse.ArgumentParser(
        description="Generate Boltz-2 YAML files from a CSV input. Supports multiple chains and complex constraints."
    )
    parser.add_argument('--input_csv', type=str,
                        help="Input CSV file path containing complex definitions.")
    parser.add_argument('--yaml_out_dir', type=str,
                        help="Output directory where YAML files will be written.")
    parser.add_argument('--csv_out', type=str,
                        help="Output CSV file path that includes the generated YAML filenames.")
    parser.add_argument('--generate_template', nargs='?', const='boltz_template.csv', default=None,
                        help="Generate a template CSV file. Optionally provide a filename.")

    args = parser.parse_args()

    # If the generate_template flag is used, create the template and exit.
    if args.generate_template:
        generate_template_csv(args.generate_template)
        return

    # Ensure required arguments are provided if not generating a template.
    if not all([args.input_csv, args.yaml_out_dir, args.csv_out]):
        parser.error("--input_csv, --yaml_out_dir, and --csv_out are required unless using --generate_template.")

    os.makedirs(args.yaml_out_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{args.input_csv}'", file=sys.stderr)
        return
        
    yaml_file_names = []
    
    for idx, row in df.iterrows():
        yaml_dict = process_row(row)
        
        # Determine the output filename
        if 'foldname' in row and not pd.isna(row['foldname']) and row['foldname'].strip():
            filename_base = row['foldname']
        else:
            # Fallback to creating a name from entity IDs if foldname is not provided
            ids = [str(row[f'entity_{i+1}_id']) for i in range(len(yaml_dict.get('sequences', [])))]
            filename_base = "_".join(ids)

        yaml_filename = f"{filename_base}.yaml"
        yaml_file_names.append(yaml_filename)
        
        yaml_path = os.path.join(args.yaml_out_dir, yaml_filename)
        with open(yaml_path, 'w') as yf:
            yaml.dump(
                yaml_dict,
                yf,
                default_flow_style=False,
                indent=2,
                sort_keys=False
            )
            
    # Add the new yaml_file column to the dataframe and save it.
    df['yaml_file'] = yaml_file_names
    df.to_csv(args.csv_out, index=False)
    
    print(f"YAML generation complete. Files are in '{args.yaml_out_dir}'.")
    print(f"Updated CSV with filenames saved to '{args.csv_out}'.")

if __name__ == '__main__':
    main()
