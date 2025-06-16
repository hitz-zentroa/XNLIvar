import pandas as pd
from sklearn.metrics import accuracy_score
import argparse
import os

def process_tsv_files(parent_dir, output_csv, is_qa=None):
    results = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".tsv"):  # Check for TSV files
                file_path = os.path.join(root, file)

                try:
                    df = pd.read_csv(file_path, sep="\t")
                    # Evaluate non-qa inference results
                    if is_qa is None: 
                        if "gold_label" in df.columns and "prediction" in df.columns:
                            y_gold = df["gold_label"].tolist()
                            y_pred = df["prediction"].tolist()
                            
                            accuracy = accuracy_score(y_gold, y_pred)
                            results.append([file_path, accuracy])
                        else:
                            print(f"Skipping {file_path}: Required columns not found.")

                    # Evaluate qa inference results
                    elif is_qa == "y":
                        for neg_value in df["prediction"].unique(): 
                            if neg_value.startswith("not_"): 
                                    evaluating_label = neg_value.replace("not_", "")

                                    # Replace all other labels in gold_label with the negated version
                                    df["gold_label"] = df["gold_label"].apply(
                                        lambda x: x if x == evaluating_label else neg_value
                                    )

                        if "gold_label" in df.columns and "prediction" in df.columns:
                            y_gold = df["gold_label"].tolist()
                            y_pred = df["prediction"].tolist()
                            
                            accuracy = accuracy_score(y_gold, y_pred)
                            results.append([file_path, accuracy])
                        else:
                            print(f"Skipping {file_path}: Required columns not found.")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Save results to a CSV
    results_df = pd.DataFrame(results, columns=["File Path", "Accuracy"])
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type=str, required=True, help="Path to the parent folder")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--is_qa", type=str, help="Are we evaluating qa inference?")
    args = parser.parse_args()
    
    process_tsv_files(args.parent_dir, args.output_csv, args.is_qa)

if __name__ == "__main__":
    main()