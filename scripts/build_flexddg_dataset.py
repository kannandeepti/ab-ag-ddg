import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

import biotite.sequence.io.fasta as fasta
from tap import tapify


def build_flexddg_dataset(json_path: Path, ddg_path: Path, save_path: Path) -> None:
    """
    Build a Flex ddG dataset by extracting antibody and antigen chain sequences
    for each mutation entry in the ddG CSV, using the provided FASTA file.

    Args:
        fasta_path (Path): Path to the FASTA file containing sequences.
        ddg_path (Path): Path to the CSV file with ddG mutation data.
        save_path (Path): Path to save the resulting CSV with sequences.
    """
    # Read the FASTA file containing all sequences
    with open(json_path, "r") as f:
        complex_to_sequence_dict = json.load(f)

    # Read the ddG CSV file into a DataFrame
    df = pd.read_csv(ddg_path)

    # Lists to store sequences for each entry
    ab_chain1 = []
    ab_chain2 = []
    ag_chains = []
    fail_count = 0

    # Iterate over each row in the DataFrame
    for row in tqdm(df.itertuples(), total=len(df)):
        complex_id = row.complex  # e.g., "7so5_HL_A_NL35G"
        # Split the complex_id into its components
        pdb_id, ab_chain, ag_chain, mutation = complex_id.split("_")

        # Parse mutation string, e.g., "NL35G"
        wt_AA = mutation[0]  # Wild-type amino acid
        chain_id = mutation[1]  # Chain identifier for the mutation
        mut_AA = mutation[-1]  # Mutant amino acid
        mut_index = complex_to_sequence_dict[f"{complex_id}_{chain_id}"][
            "mutation_index"
        ]
        sequence_of_mutated_chain = complex_to_sequence_dict[
            f"{complex_id}_{chain_id}"
        ]["sequence"]
        # Check that the wild-type amino acid matches the sequence at the mutation position
        if sequence_of_mutated_chain[mut_index] != wt_AA:
            fail_count += 1  # For debugging if mismatch occurs

        # Retrieve the sequences for the chain where the mutation occurs
        ab_chain1.append(
            complex_to_sequence_dict[f"{complex_id}_{ab_chain[0]}"]["sequence"]
        )
        if len(ab_chain) > 1:
            ab_chain2.append(
                complex_to_sequence_dict[f"{complex_id}_{ab_chain[1]}"]["sequence"]
            )
        else:
            ab_chain2.append("")
        ag_chains.append(
            complex_to_sequence_dict[f"{complex_id}_{ag_chain}"]["sequence"]
        )

    df["ab_chain1_seq"] = ab_chain1
    df["ab_chain2_seq"] = ab_chain2
    df["ag_chain_seq"] = ag_chains

    # Save the updated DataFrame to CSV
    df.to_csv(save_path, index=False)
    print(f"Failed to match {fail_count} sequences")


if __name__ == "__main__":
    tapify(build_flexddg_dataset)
