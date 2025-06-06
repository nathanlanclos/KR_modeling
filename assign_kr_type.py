import pandas as pd

def assign_core_kr_type(annotation_string):
    """
    Assigns a core KR type only if it is explicitly stated in the annotation string.
    This function avoids making inferences based on downstream domains (DH, ER)
    as their activity modifies the substrate, obscuring the original KR product.
    """
    if pd.isna(annotation_string):
        annotation_string = ""
    else:
        annotation_string = str(annotation_string)

    # Define explicit KR type identifiers, checking for specific subtypes first
    # to ensure the most precise classification is captured.
    specific_kr_subtypes = [
        "A0KR", "A1KR", "A2KR", "B0KR", "B1KR", "B2KR",
        "C0KR", "C1KR", "C2KR"
    ]
    general_kr_types = ["AKR", "BKR", "CKR"]

    # 1. Check for specific KR subtypes (e.g., A1KR)
    for kr_id in specific_kr_subtypes:
        if kr_id in annotation_string:
            rationale = f"Directly identified specific subtype '{kr_id}' in annotation string."
            return kr_id, rationale

    # 2. Check for general KR types (e.g., AKR)
    for kr_id in general_kr_types:
        if kr_id in annotation_string:
            rationale = f"Directly identified general type '{kr_id}' in annotation string."
            return kr_id, rationale

    # 3. Handle all other cases where no explicit KR type is found
    if not annotation_string or annotation_string.lower() == "nan":
        rationale = "Annotation string is empty or NaN."
    else:
        rationale = "No explicit KR type found in the annotation string. Type cannot be inferred due to potential downstream modifications."

    return "Undetermined", rationale

# --- Main script execution block ---
if __name__ == "__main__":
    print("Starting KR type assignment script (Direct Annotation Only)...")
    
    # Get CSV file paths from user
    csv_file_path = input("Enter the path to your input CSV file: ")
    output_csv_file_path = input("Enter the path for the output CSV file (e.g., processed_kr_data.csv): ")

    try:
        print(f"Reading CSV file from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print("CSV file read successfully.")

        # Ensure the 'Annotation' column exists
        if "Annotation" not in df.columns:
            print("Error: The CSV file must contain an 'Annotation' column.")
        else:
            print("Processing annotations...")
            # Apply the assignment function
            results = df["Annotation"].apply(assign_core_kr_type)
            
            # Create the new columns from the results
            df["core_kr_type"] = [result[0] for result in results]
            df["assignment_rationale"] = [result[1] for result in results]
            
            print(f"Saving processed data to: {output_csv_file_path}")
            # Save the updated DataFrame to a new CSV file
            df.to_csv(output_csv_file_path, index=False)
            print(f"Processing complete. Output saved to '{output_csv_file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Please check the path and try again.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file_path}' is empty. Please provide a CSV with data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")