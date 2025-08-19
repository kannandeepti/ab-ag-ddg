"""Generate likelihood ratios of mutant vs wildtype antigens using the ESM2 model from https://github.com/facebookresearch/esm.
Uses the masked marginals scoring method from https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full.
Code adapted from https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py.
"""

from pathlib import Path
from typing import Literal

from Bio.SeqIO.Interfaces import SequentialSequenceWriter
import pandas as pd
import torch
from tap import tapify
from tqdm import tqdm, trange

from esm_embedding import load_esm_model

# Amino acid constants
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_ALPHABET_SET = set(AA_ALPHABET)
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ALPHABET)}
GLYCINE_LINKER_LENGTH = 25


def parse_csv_with_sequences(
    csv_path: Path, sequence_type: Literal["single_chain", "triple_chain"]
) -> tuple[list[str], list[str]]:
    """
    Parse a CSV file with sequences and extract sequence descriptions and sequences.
    """
    df = pd.read_csv(csv_path)

    names, sequences, mut_indices, wt_aas, mut_aas = [], [], [], [], []
    for row in df.itertuples():
        pdb_id, ab_chain, ag_chain, mutation = row.complex.split("_")
        wt_aas.append(mutation[0])
        mut_aas.append(mutation[-1])

        if sequence_type == "single_chain":
            if row.mut_chain == ab_chain[0]:
                sequence = row.ab_chain1_seq
            elif row.mut_chain == ag_chain:
                sequence = row.ag_chain_seq
            elif row.mut_chain == ab_chain[1]:
                sequence = row.ab_chain2_seq
            else:
                raise ValueError(f"Invalid mutation chain: {row.mut_chain}")
            sequences.append(sequence)
            names.append(f"{row.complex}_{row.mut_chain}")
            mut_indices.append(row.mut_index)

        elif sequence_type == "triple_chain":
            if row.mut_chain == ag_chain:
                mut_index = row.mut_index
            elif row.mut_chain == ab_chain[0]:
                mut_index = (
                    len(row.ag_chain_seq) + GLYCINE_LINKER_LENGTH + row.mut_index
                )
            elif row.mut_chain == ab_chain[1]:
                mut_index = (
                    len(row.ag_chain_seq)
                    + len(row.ab_chain1_seq)
                    + 2 * GLYCINE_LINKER_LENGTH
                    + row.mut_index
                )
            else:
                raise ValueError(f"Invalid mutation chain: {row.mut_chain}")
            mut_indices.append(mut_index)
            sequence = row.ag_chain_seq
            sequence += "G" * GLYCINE_LINKER_LENGTH
            sequence += row.ab_chain1_seq
            if len(ab_chain) == 2:
                sequence += "G" * GLYCINE_LINKER_LENGTH
                sequence += row.ab_chain2_seq
            sequences.append(sequence)
            names.append(f"{row.complex}")
        else:
            raise ValueError(f"Invalid sequence type: {sequence_type}")

    return names, sequences, mut_indices, wt_aas, mut_aas


def generate_likelihood_ratios(
    sequences_path: Path,
    sequence_type: Literal["single_chain", "triple_chain"],
    batch_size: int = 32,
    device: str = "cuda",
) -> None:
    """Generate likelihood ratios of mutant vs wildtype antigens using the ESM2 model from https://github.com/facebookresearch/esm.

    :param save_path: Path to PT file where a dictionary mapping antigen name to likelihood will be saved.
    :param device: The device to use (e.g., "cpu" or "cuda") for the model.
    """
    # Load ESM-2 model
    model, alphabet, batch_converter = load_esm_model()
    names, sequences, mut_indices, wt_aas, mut_aas = parse_csv_with_sequences(
        sequences_path, sequence_type
    )
    sequence_tuples = list(zip(names, sequences))

    # Move model to device
    model = model.to(device)

    name_to_likelihood_ratio = {}

    with torch.no_grad():
        # Iterate over batches of sequences
        for i in trange(0, len(sequence_tuples), batch_size):
            # Get batch of sequences
            batch_sequence_tuples = sequence_tuples[i : i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(
                batch_sequence_tuples
            )
            batch_mut_indices = torch.tensor(mut_indices[i : i + batch_size])
            batch_wt_aas = torch.tensor(
                [alphabet.get_idx(wt_aas[k]) for k in range(i, i + batch_size)]
            )
            batch_mut_aas = torch.tensor(
                [alphabet.get_idx(mut_aas[k]) for k in range(i, i + batch_size)]
            )
            breakpoint()
            batch_tokens[:, batch_mut_indices] = alphabet.mask_idx

            batch_token_probs = torch.log_softmax(
                model(batch_tokens.to(device))["logits"], dim=-1
            )  # shape: (batch_size, seq_len, vocab_size)
            batch_token_probs = batch_token_probs[
                :, 1:, :
            ]  # remove beginning-of-sequence token

            # For each row in the batch, extract the log-probability at the mutation index for the WT and mutant amino acid
            batch_indices = torch.arange(batch_tokens.size(0))
            batch_likelihood_ratios = (
                batch_token_probs[batch_indices, batch_mut_indices, batch_wt_aas]
                - batch_token_probs[batch_indices, batch_mut_indices, batch_mut_aas]
            )  # shape: (batch_size,)
            batch_likelihood_ratios = batch_likelihood_ratios.cpu()

            for name, likelihood_ratio in zip(batch_labels, batch_likelihood_ratios):
                name_to_likelihood_ratio[name] = likelihood_ratio.item()

    return name_to_likelihood_ratio


if __name__ == "__main__":

    tapify(generate_likelihood_ratios)
