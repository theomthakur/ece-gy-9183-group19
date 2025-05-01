import os
import shutil
import random

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
OUTPUT_BASE_DIR = "data/organized"

def main():

    splits = {
        "training": {"labeled": 11250, "unlabeled": 0},
        "validation": {"labeled": 750, "unlabeled": 0},
        "test": {"labeled": 750, "unlabeled": 0},
        "staging": {"labeled": 1000, "unlabeled": 1000},
        "canary": {"labeled": 500, "unlabeled": 1000},
        "production": {"labeled": 500, "unlabeled": 1000}
    }

    flat_structure_splits = ["training", "validation", "test"]
    subdir_splits = ["staging", "canary", "production"]

    for split_name in splits.keys():
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, split_name), exist_ok=True)

    if split_name in subdir_splits:
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, split_name, "labeled"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, split_name, "unlabeled"), exist_ok=True)

    labeled_files = []
    for file in os.listdir(TRAIN_DIR):
        if os.path.isfile(os.path.join(TRAIN_DIR, file)):
            labeled_files.append(file)

    random.shuffle(labeled_files)

    labeled_used = 0
    for split_name, counts in splits.items():
        labeled_count = counts["labeled"]
        if labeled_count > 0:
            split_files = labeled_files[labeled_used:labeled_used + labeled_count]
            labeled_used += labeled_count
            
            for file in split_files:
                src = os.path.join(TRAIN_DIR, file)
                
                if split_name in flat_structure_splits:
                    dst = os.path.join(OUTPUT_BASE_DIR, split_name, file)
                else:
                    dst = os.path.join(OUTPUT_BASE_DIR, split_name, "labeled", file)
                    
                shutil.copy(src, dst)

    unlabeled_files = []
    for file in os.listdir(TEST_DIR):
        if os.path.isfile(os.path.join(TEST_DIR, file)):
            unlabeled_files.append(file)

    random.shuffle(unlabeled_files)

    unlabeled_used = 0
    for split_name, counts in splits.items():
        unlabeled_count = counts["unlabeled"]
        if unlabeled_count > 0:
            split_files = unlabeled_files[unlabeled_used:unlabeled_used + unlabeled_count]
            unlabeled_used += unlabeled_count
            
            for file in split_files:
                src = os.path.join(TEST_DIR, file)
                dst = os.path.join(OUTPUT_BASE_DIR, split_name, "unlabeled", file)
                shutil.copy(src, dst)

    print(f"Data organization complete. Used {labeled_used} labeled files and {unlabeled_used} unlabeled files.")

if __name__ == "__main__":
    main()