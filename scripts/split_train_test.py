# Create separate train, val test files for training

import pandas as pd
from pathlib import Path


def transfer_columns_to_csv(
    ddg_with_sequences_path: Path,
    ddg_path: Path,
) -> None:
    ddg_with_sequences = pd.read_csv(ddg_with_sequences_path)
    ddg_without_sequences = pd.read_csv(ddg_path)

    # rename complex column in ddg_without_sequences to match the complex column in ddg_with_sequences
    name_to_split = {}
    for row in ddg_without_sequences.itertuples():
        name = row.complex
        complex_id, mutation = name.split("_")
        ab_chain = row.ab_chain
        ag_chain = row.ag_chain
        name_to_split[f"{complex_id}_{ab_chain}_{ag_chain}_{mutation}"] = row.split

    # add name_to_split dictionary as a column with title "split"
    ddg_with_sequences["split"] = ddg_with_sequences["complex"].map(name_to_split)
    df = ddg_with_sequences
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    train_df.to_csv(ddg_path.parent / f"{ddg_path.stem}_train.csv", index=False)
    val_df.to_csv(ddg_path.parent / f"{ddg_path.stem}_val.csv", index=False)
    test_df.to_csv(ddg_path.parent / f"{ddg_path.stem}_test.csv", index=False)


if __name__ == "__main__":
    for split_type in ["70", "90", "100", "none"]:
        transfer_columns_to_csv(
            Path(
                "ddg_synthetic/Flex_ddG/Synthetic_FlexddG_ddG_20829_with_sequences.csv"
            ),
            Path(
                f"ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_{split_type}.csv"
            ),
        )
