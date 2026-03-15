"""
Step 0: Download UltraDomain dataset from HuggingFace and extract unique contexts.

Downloads JSONL files from https://huggingface.co/datasets/TommyChien/UltraDomain
and deduplicates the "context" field from each file.

Usage:
    python Step_0.py                          # Download + extract all domains
    python Step_0.py -d agriculture cs legal  # Only specific domains
    python Step_0.py --skip-download          # Skip download, just extract
"""

import os
import json
import glob
import argparse

# All available domains in the UltraDomain dataset
ALL_DOMAINS = [
    "agriculture",
    "art",
    "biography",
    "biology",
    "cooking",
    "cs",
    "fiction",
    "fin",
    "health",
    "history",
    "legal",
    "literature",
    "mathematics",
    "mix",
    "music",
    "philosophy",
    "physics",
    "politics",
    "psychology",
    "technology",
]

DATASET_REPO = "TommyChien/UltraDomain"


def download_datasets(domains, output_directory):
    """Download JSONL files from HuggingFace for the specified domains."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess

        subprocess.check_call(["pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    os.makedirs(output_directory, exist_ok=True)

    for domain in domains:
        filename = f"{domain}.jsonl"
        dest_path = os.path.join(output_directory, filename)

        if os.path.exists(dest_path):
            print(f"Already downloaded: {filename}")
            continue

        print(f"Downloading {filename} from {DATASET_REPO}...")
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=output_directory,
        )
        print(f"  Saved to: {dest_path}")

    print("Download complete.")


def extract_unique_contexts(input_directory, output_directory, domains=None):
    """Extract unique contexts from JSONL files."""
    os.makedirs(output_directory, exist_ok=True)

    if domains:
        jsonl_files = [
            os.path.join(input_directory, f"{d}.jsonl")
            for d in domains
            if os.path.exists(os.path.join(input_directory, f"{d}.jsonl"))
        ]
    else:
        jsonl_files = glob.glob(os.path.join(input_directory, "*.jsonl"))

    print(f"Found {len(jsonl_files)} JSONL files.")

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_unique_contexts.json"
        output_path = os.path.join(output_directory, output_filename)

        unique_contexts_dict = {}

        print(f"Processing file: {filename}")

        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        context = json_obj.get("context")
                        if context and context not in unique_contexts_dict:
                            unique_contexts_dict[context] = None
                    except json.JSONDecodeError as e:
                        print(
                            f"JSON decoding error in file {filename} at line {line_number}: {e}"
                        )
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue
        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")
            continue

        unique_contexts_list = list(unique_contexts_dict.keys())
        print(
            f"There are {len(unique_contexts_list)} unique `context` entries in the file {filename}."
        )

        try:
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(unique_contexts_list, outfile, ensure_ascii=False, indent=4)
            print(f"Unique `context` entries have been saved to: {output_filename}")
        except Exception as e:
            print(f"An error occurred while saving to the file {output_filename}: {e}")

    print("All files have been processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download UltraDomain dataset and extract unique contexts"
    )
    parser.add_argument("-i", "--input_dir", type=str, default="../datasets")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="../datasets/unique_contexts"
    )
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        default=["agriculture", "cs", "legal", "mix"],
        help=f"Domains to process (default: agriculture cs legal mix). "
        f"Available: {', '.join(ALL_DOMAINS)}",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, only extract from existing files",
    )

    args = parser.parse_args()

    if not args.skip_download:
        download_datasets(args.domains, args.input_dir)

    extract_unique_contexts(args.input_dir, args.output_dir, args.domains)
