# Create separate train, val test files for training

import pandas as pd
from pathlib import Path


def split_train_test(ddg_csv_path: Path) -> None:
    df = pd.read_csv(ddg_csv_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    train_df.to_csv(ddg_csv_path.parent / f"{ddg_csv_path.stem}_train.csv", index=False)
    val_df.to_csv(ddg_csv_path.parent / f"{ddg_csv_path.stem}_val.csv", index=False)
    test_df.to_csv(ddg_csv_path.parent / f"{ddg_csv_path.stem}_test.csv", index=False)


if __name__ == "__main__":
    for csv_file in Path("ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs").glob("*.csv"):
        split_train_test(csv_file)
