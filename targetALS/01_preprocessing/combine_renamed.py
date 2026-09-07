import pandas as pd
import glob
import os
import argparse

def combine_files(folder_path, output_filename="combined_union.csv"):
    """
    Finds all *_renamed.csv files in a folder and merges them 
    on 'code' and 'group' columns.
    """
    # 1. Search for renamed files
    search_pattern = os.path.join(folder_path, "*_renamed.csv")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"Error: No files matching '*_renamed.csv' found in {folder_path}")
        return

    print(f"Found {len(files)} files to combine.")

    combined_df = None

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"  Reading: {filename}...")
        
        df = pd.read_csv(file_path)

        if combined_df is None:
            # First file establishes the base
            combined_df = df
        else:
            # Merge following files on common keys
            # 'how=outer' ensures no data is lost if codes differ between files
            combined_df = pd.merge(combined_df, df, on=['code', 'group'], how='outer')

    # 2. Save the final result
    output_path = os.path.join(folder_path, output_filename)
    combined_df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"SUCCESS: Union file created at: {output_path}")
    print(f"Final dimensions: {combined_df.shape[0]} rows x {combined_df.shape[1]} columns")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple *_renamed.csv files into one union file.")
    parser.add_argument("folder", type=str, help="The path to the folder containing the _renamed.csv files.")
    parser.add_argument("--output", type=str, default="combined_union.csv", help="Optional: Name of the output file.")

    args = parser.parse_args()
    combine_files(args.folder, args.output)