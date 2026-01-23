#!/usr/bin/env python3
"""
Merge Output Files Script

Concatenates all Task A and Task C output files into single submission files.
Also fixes task_id format from :: to <::> to match qrels format.
"""

import argparse
import json
import os
from pathlib import Path

# Dataset names
DATASETS = ["clapnq", "fiqa", "govt", "ibmcloud"]


def fix_task_id(task_id: str) -> str:
    """Convert task_id format from :: to <::> to match qrels format."""
    if task_id and "::" in task_id and "<::>" not in task_id:
        return task_id.replace("::", "<::>")
    return task_id


def merge_files(input_dir: str, output_file: str, task_type: str, fix_ids: bool = True):
    """
    Merge all dataset files for a given task type.

    Args:
        input_dir: Directory containing the input files
        output_file: Path to the output merged file
        task_type: Either 'task_a' or 'task_c'
        fix_ids: Whether to fix task_id format (:: -> <::>)
    """
    all_records = []

    for dataset in DATASETS:
        input_file = os.path.join(input_dir, f"{task_type}_{dataset}.jsonl")

        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue

        count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)

                # Fix task_id format if needed
                if fix_ids and 'task_id' in record:
                    record['task_id'] = fix_task_id(record['task_id'])

                all_records.append(record)
                count += 1

        print(f"Loaded {count} records from {dataset}")

    # Write merged output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record) + '\n')

    print(f"\nMerged {len(all_records)} total records into {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge Task A and Task C output files into submission files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output',
        help='Directory containing the task output files (default: output)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/submission',
        help='Directory for merged submission files (default: output/submission)'
    )
    parser.add_argument(
        '--team-name',
        type=str,
        default='NLP-CEIA-UFG',
        help='Team name for output files (default: NLP-CEIA-UFG)'
    )
    parser.add_argument(
        '--no-fix-ids',
        action='store_true',
        help='Do not fix task_id format (keep :: instead of converting to <::>)'
    )

    args = parser.parse_args()

    fix_ids = not args.no_fix_ids

    print("=" * 60)
    print("Merging Task A files...")
    print("=" * 60)
    task_a_output = os.path.join(args.output_dir, f"{args.team_name}_taskA.jsonl")
    merge_files(args.input_dir, task_a_output, "task_a", fix_ids=fix_ids)

    print("\n" + "=" * 60)
    print("Merging Task C files...")
    print("=" * 60)
    task_c_output = os.path.join(args.output_dir, f"{args.team_name}_taskC.jsonl")
    merge_files(args.input_dir, task_c_output, "task_c", fix_ids=fix_ids)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Task A submission: {task_a_output}")
    print(f"Task C submission: {task_c_output}")


if __name__ == '__main__':
    main()