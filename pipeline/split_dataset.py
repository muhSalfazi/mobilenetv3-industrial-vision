import os
import shutil
import random

def split_dataset(source_dir, out_dir, split_ratios={"train": 0.7, "val": 0.2, "test": 0.1}):
    classes = os.listdir(source_dir)

    for split in split_ratios:
        for cls in classes:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        files = os.listdir(cls_path)
        random.shuffle(files)

        n = len(files)
        n_train = int(n * split_ratios["train"])
        n_val = int(n * split_ratios["val"])

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }

        for split, file_list in splits.items():
            for f in file_list:
                shutil.copy(
                    os.path.join(cls_path, f),
                    os.path.join(out_dir, split, cls, f)
                )

    print("[DONE] Dataset split into train/val/test.")
