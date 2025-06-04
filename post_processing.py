#!/usr/bin/env python3
import pandas as pd
import os
import re
import json
from collections import Counter
import argparse
from tqdm import tqdm
from pathlib import Path
import glob

def extract_mcq_answer(text):
    preprocessed_text = preprocess_text(text)
    preprocessed_text = preprocessed_text.upper()
    all_matches = []

    first_match = re.match(r'^([ABCD])(?:\.|:|\)|\s|$)', preprocessed_text)
    if first_match:
        match = first_match
        all_matches.append((match.start(), match.group(1), match.group(0)))
        return all_matches[-1][1], all_matches[-1][2]
    first_word_match = re.match(r'^([ABCD])\s+(?:EST|SERAIT)', preprocessed_text, re.IGNORECASE)
    if first_word_match:
        match = first_word_match
        all_matches.append((match.start(), match.group(1), match.group(0)))
        return all_matches[-1][1], all_matches[-1][2]

    patterns = [
        r'(?:ANSWER|ANS|FINAL ANSWER|SOLUTION)(?:\s+IS)?[:=\s]+([ABCD])\b',
        r'(?:I|WE)(?:\s+CHOOSE|SELECT|PICK|GO WITH|CONCLUDE|THINK)(?:\s+(?:OPTION|CHOICE|ANSWER))?[:=\s]+([ABCD])\b',
        r'(?:OPTION|CHOICE|ANSWER)[:=\s]+([ABCD])\b',
        r'(?:THE\s+)?(?:CORRECT|RIGHT)(?:\s+(?:OPTION|CHOICE|ANSWER))?(?:\s+IS)?[:=\s]+([ABCD])\b',
        r'([ABCD])\s+(?:IS|WOULD BE)(?:\s+THE)(?:\s+(?:CORRECT|RIGHT))?(?:\s+(?:OPTION|CHOICE|ANSWER))?',
        r'([ABCD])\s+(?:IS|WOULD BE)(?:\s+(?:CORRECT|RIGHT))?',
        r'THEREFORE,? (?:THE )?(?:ANSWER|OPTION|CHOICE) (?:IS|WOULD BE)[:=\s]+([ABCD])\b',
        r'(?:EST )[:=\s]+([ABCD])\b',
        r'(?:RÉPONSE|RÉP|RÉPONSE FINALE|SOLUTION)(?:\s+EST)?[:=\s]+([ABCD])\b',
        r'(?:RÉPONSE|RÉP|RÉPONSE FINALE|SOLUTION)[\s:=\s]+([ABCD])\b',
        r'(?:JE|NOUS)(?:\s+CHOISIS|CHOISISSONS|SÉLECTIONNE|SÉLECTIONNONS|OPTE|OPTONS POUR|CONCLU|CONCLUONS|PENSE|PENSONS)(?:\s+(?:OPTION|CHOIX|RÉPONSE))?[:=\s]+([ABCD])\b',
        r'(?:OPTION|CHOIX|RÉPONSE)[:=\s]+([ABCD])\b',
        r'(?:LA\s+)?(?:CORRECTE|BONNE)(?:\s+(?:OPTION|CHOIX|RÉPONSE))?(?:\s+EST)?[:=\s]+([ABCD])\b',
        r'(?:LA\s+)?(?:CORRECTE|BONNE)(?:\s+(?:OPTION|CHOIX|RÉPONSE))?(?:\s+DEVRAIT)?(?:\s+ÊTRE)?[:=\s]+([ABCD])\b',
        r'(?:LA\s+)?(?:OPTION|CHOIX|RÉPONSE)(?:\s+(?:CORRECTE|BONNE))?(?:\s+EST)?(?:\s+DONC)?[:=\s]+([ABCD])\b',
        r'([ABCD])\s+(?:EST|SERAIT)(?:\s+LA)(?:\s+(?:CORRECTE|BONNE))?(?:\s+(?:OPTION|CHOIX|RÉPONSE))?',
        r'([ABCD])\s+(?:EST|SERAIT)(?:\s+(?:CORRECTE|BONNE))?',
        r'PAR CONSÉQUENT,? (?:LA )?(?:RÉPONSE|OPTION|CHOIX) (?:EST|SERAIT)[:=\s]+([ABCD])\b',
        r'(?:EST|SERAIT )[:=\s]+([ABCD])\b',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, preprocessed_text)
        for match in matches:
            all_matches.append((match.start(), match.group(1), match.group(0)))

    standalone_patterns = [
        r'^([ABCD])$',
        r'^(?:ANSWER:?)?\s*([ABCD])$',
        r'\s(?:ANSWER:?)?\s*([ABCD])$',
    ]

    for pattern in standalone_patterns:
        matches = re.finditer(pattern, preprocessed_text)
        for match in matches:
            all_matches.append((match.start(), match.group(1), match.group(0)))

    if all_matches:
        all_matches.sort()
        return all_matches[-1][1], all_matches[-1][2]

    words = preprocessed_text.split()
    if words and words[-1] in ["A", "B", "C", "D"]:
        return words[-1], words[-1]

    all_options = []
    for match in re.finditer(r'\b[ABCD]\b', preprocessed_text):
        all_options.append((match.start(), match.group(), match.group()))

    if all_options:
        all_options.sort()
        return all_options[-1][1], all_options[-1][2]

    return None, None

def preprocess_text(raw_text):
    text = re.sub(r'(\n|\\n)', '', raw_text)
    text = re.sub(r'\box\{([ABCD])\}', '', text)
    text = re.sub(r'\\', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\["?|"?\]$', '', text).strip()
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```(?:.*?\n)?(.*?)```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^"(.*)"$', r'\1', text)
    text = re.sub(r"^'(.*)'$", r'\1', text)
    text = re.sub(r'^«\s*(.*)\s*»$', r'\1', text)
    text = re.sub(r'\\boxed\{\\text\{([A-D])\.(.*?)\}\s*\\}', r'\1', text)
    text = re.sub(r'\\boxed{([A-D])\.', r'\1', text)
    text = re.sub(r'\\\[.*?\\\]|\\\(.*?\\\)', '', text, flags=re.DOTALL)
    text = re.sub(r'\.\n\nRéponse: [A-Z]\. [a-z]$', '', text)
    return text.strip()

def compute_new_metric(prediction, gold):
    pred_answer, _ = extract_mcq_answer(prediction)
    correct = pred_answer == gold
    found = pred_answer is not None
    return int(correct), found

def process_parquet_file(file_path):
    df = pd.read_parquet(file_path)

    # Initialize metrics column if not present
    if 'metrics' not in df.columns:
        df['metrics'] = [{}] * len(df)

    # Convert any non-dict metrics to dict
    for idx in range(len(df)):
        if not isinstance(df.at[idx, 'metrics'], dict):
            df.at[idx, 'metrics'] = {}

    # Compute metrics
    correct_count = 0
    found_count = 0
    total_count = len(df)

    for idx, row in df.iterrows():
        prediction = str(row.get("predictions", ""))
        gold = str(row.get("gold", ""))

        # Extract gold answer character (assuming it's in format like "'A'" or similar)
        if len(gold) >= 3:
            gold_answer = gold[2] if gold[1] == "'" and gold[3] == "'" else gold[0]
        else:
            gold_answer = gold[0] if gold else ""

        new_metric_value, found = compute_new_metric(prediction, gold_answer)
        df.at[idx, 'metrics']['new'] = new_metric_value

        if new_metric_value:
            correct_count += 1
        if found:
            found_count += 1

    # Calculate accuracy metrics
    base_correct = sum(row.get("metrics", {}).get("acc", 0) for _, row in df.iterrows())

    # Adjusted accuracy (scaling from [0.25, 1.0] to [0, 1.0])
    base_adjusted = (base_correct/total_count-0.25)/0.75*100 if total_count > 0 else 0
    new_adjusted = (correct_count/total_count-0.25)/0.75*100 if total_count > 0 else 0

    results = {
        'total': total_count,
        'base_correct': base_correct,
        'base_accuracy': base_correct/total_count*100 if total_count > 0 else 0,
        'base_adjusted': base_adjusted,
        'new_correct': correct_count,
        'new_accuracy': correct_count/total_count*100 if total_count > 0 else 0,
        'new_adjusted': new_adjusted,
        'found': found_count,
        'not_found': total_count - found_count
    }

    # Optionally save back to file
    # df.to_parquet(file_path)

    return results

def find_and_process_parquet_files(directory, scores_directory=None, save_results=False):
    print(f"Scanning directory: {directory}")
    results = {}

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(directory):
        # Filter for parquet files containing "gpqa" in their names
        parquet_files = [f for f in files if f.endswith('.parquet') and 'gpqa' in f.lower()]

        if parquet_files:
            for file in tqdm(parquet_files, desc="Processing files"):
                file_path = os.path.join(root, file)
                try:
                    print(f"\nProcessing file: {file_path}")
                    file_results = process_parquet_file(file_path)
                    results[file_path] = file_results

                    # Print results for this file
                    print(f"  Total examples: {file_results['total']}")
                    print(f"  Base accuracy: {file_results['base_correct']}/{file_results['total']} ({file_results['base_accuracy']:.2f}%)")
                    print(f"  Base adjusted: {file_results['base_adjusted']:.2f}%")
                    print(f"  New accuracy: {file_results['new_correct']}/{file_results['total']} ({file_results['new_accuracy']:.2f}%)")
                    print(f"  New adjusted: {file_results['new_adjusted']:.2f}%")
                    print(f"  Not found: {file_results['not_found']}/{file_results['total']} ({file_results['not_found']/file_results['total']*100:.2f}%)")

                    # Update corresponding JSON in scores directory if provided
                    if scores_directory:
                        update_scores_json(file_path, file_results, directory, scores_directory)

                    if save_results:
                        output_path = file_path.replace('.parquet', '_metrics.csv')
                        pd.DataFrame([file_results]).to_csv(output_path, index=False)
                        print(f"  Results saved to: {output_path}")

                except Exception as e:
                    print(f"  Error processing {file_path}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())

    return results

def update_scores_json(parquet_path, file_results, parquets_dir, scores_dir):
    """
    Update the corresponding JSON file in the scores directory with new accuracy metrics.

    Args:
        parquet_path: Path to the parquet file that was processed
        file_results: Dictionary containing the calculated metrics
        parquets_dir: Base directory for parquet files
        scores_dir: Base directory for score files
    """
    try:
        # Get relative path from parquets directory
        rel_path = os.path.relpath(parquet_path, parquets_dir)
        parent_dirs = os.path.dirname(rel_path)

        # Split the path into components
        path_parts = Path(rel_path).parts

        # Find matching results JSON file (looking for results_YYYY-MM-DD*.json)
        potential_json_dir = os.path.join(scores_dir, path_parts[1], path_parts[2])
        json_pattern = os.path.join(potential_json_dir, "results_*.json")
        json_files = glob.glob(json_pattern)

        if not json_files:
            print(f"  No matching JSON files found at {json_pattern}")
            return

        # Use the most recent JSON file if multiple exist
        json_file = sorted(json_files)[-1]
        print(f"  Found corresponding JSON file: {json_file}")

        # Read the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract dataset name from parquet filename or path
        filename = os.path.basename(parquet_path)
        if 'gpqa-fr' in filename.lower() or 'gpqa-fr' in parquet_path.lower():
            dataset_key = "community|gpqa-fr|0"
        else:
            dataset_key = "community|gpqa|0"  # Default key if not specified

        # Update the JSON with new accuracy
        new_accuracy = file_results['new_accuracy']/100.0

        # Create necessary structure if it doesn't exist
        if 'results' not in data:
            data['results'] = {}
        if dataset_key not in data['results']:
            data['results'][dataset_key] = {}

        # Add new accuracy metric
        data['results'][dataset_key]['new_acc'] = new_accuracy

        # Write back to the JSON file
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Updated JSON file with new_acc: {new_accuracy:.2f}%")

    except Exception as e:
        print(f"  Error updating JSON file: {str(e)}")
        import traceback
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description='Compute metrics for parquet files containing "gpqa" in their names.')
    parser.add_argument('directory', help='The directory to scan for parquet files')
    parser.add_argument('--scores-dir', help='The directory containing corresponding score JSON files', required=True)
    parser.add_argument('--save', action='store_true', help='Save results to CSV files')
    args = parser.parse_args()

    find_and_process_parquet_files(args.directory, args.scores_dir, args.save)

if __name__ == "__main__":
    main()
