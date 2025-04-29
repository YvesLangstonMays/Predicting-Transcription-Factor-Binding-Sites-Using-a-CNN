import os
import numpy as np

# Setup project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_small")

TFs = ["CEBPA", "CTCF", "GATA1", "TP53"]
SUBSET_SIZES = {
    "train": 500,
    "val": 50,
    "test": 50
}

def safe_load_and_slice(path, count):
    try:
        data = np.load(path)
        return data[:min(count, len(data))]
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None

for tf in TFs:
    input_path = os.path.join(INPUT_DIR, tf)
    output_path = os.path.join(OUTPUT_DIR, tf)
    os.makedirs(output_path, exist_ok=True)

    for split in ["train", "val", "test"]:
        x_file = os.path.join(input_path, f"X_{split}.npy")
        y_file = os.path.join(input_path, f"y_{split}.npy")

        x_subset = safe_load_and_slice(x_file, SUBSET_SIZES[split])
        y_subset = safe_load_and_slice(y_file, SUBSET_SIZES[split])

        if x_subset is not None:
            np.save(os.path.join(output_path, f"X_{split}.npy"), x_subset)
        if y_subset is not None:
            np.save(os.path.join(output_path, f"y_{split}.npy"), y_subset)

    # Copy metadata.json if exists
    meta_src = os.path.join(input_path, "metadata.json")
    meta_dst = os.path.join(output_path, "metadata.json")
    if os.path.exists(meta_src):
        import shutil
        shutil.copy(meta_src, meta_dst)

    print(f"[✓] Saved subset for {tf} to {output_path}")

print("\nAll TFs processed.")