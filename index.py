import json
import os
import glob

def clean_notebook(file_path):
    print(f"Processing: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        # 1. Clean top-level metadata
        # Keep only essential fields if they exist
        essential_metadata = {}
        if "metadata" in nb:
            for key in ["kernelspec", "language_info"]:
                if key in nb["metadata"]:
                    essential_metadata[key] = nb["metadata"][key]
        nb["metadata"] = essential_metadata
        
        # 2. Clean cell-level metadata
        if "cells" in nb:
            for cell in nb["cells"]:
                # Remove cell metadata
                cell["metadata"] = {}
                
                # Optionally clear outputs and execution counts for code cells
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None

        # Save the cleaned notebook (overwriting or creating a new one)
        # Here we'll create a 'cleaned_' version to be safe, or just overwrite if preferred.
        # The user asked to "remove meta data", so I'll overwrite to fulfill the request directly.
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        
        print(f"Successfully cleaned: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Find all .ipynb files in the current directory
    notebooks = glob.glob("DPO_Fine_Tuning_v1_(1).ipynb")
    
    if not notebooks:
        print("No .ipynb files found in the current directory.")
    else:
        for nb_file in notebooks:
            clean_notebook(nb_file)
        print("\nAll notebooks processed.")
