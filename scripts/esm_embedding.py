"""Generate antibody embeddings using the ESM2 model from https://github.com/facebookresearch/esm."""

from time import time
from pathlib import Path
import pandas as pd
from typing import Literal
from tap import tapify
import torch
from Bio import SeqIO
from tap import Tap
from tqdm import trange

from utils import GLYCINE_LINKER_LENGTH, load_esm_model


def mutate_sequence(sequence: str, mut_index: int, mut_aa: str) -> str:
    """Mutate a sequence at a given index with a given amino acid."""
    if mut_index == 0:
        return mut_aa + sequence[1:]
    elif mut_index == len(sequence) - 1:
        return sequence[:-1] + mut_aa
    else:
        return sequence[:mut_index] + mut_aa + sequence[mut_index + 1 :]


def parse_and_mutate_sequences(
    df: pd.DataFrame,
    sequence_type: Literal["separate_chains", "joined_chains"],
    mutate: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Parse a CSV file with sequences and extract sequence descriptions and sequences.
    """
    if sequence_type == "separate_chains":
        (
            names_ab1,
            names_ab2,
            names_ag,
            ab_chain_1_sequences,
            ab_chain_2_sequences,
            ag_chain_sequences,
        ) = ([], [], [], [], [], [])
        for row in df.itertuples():
            pdb_id, ab_chain, ag_chain, mutation = row.complex.split("_")
            mut_aa = mutation[-1]
            names_ab1.append(f"{row.complex}_{ab_chain[0]}")
            names_ag.append(f"{row.complex}_{ag_chain}")

            if row.mut_chain == ag_chain and mutate:
                ag_chain_sequences.append(
                    mutate_sequence(row.ag_chain_seq, row.mut_index, mut_aa)
                )
            else:
                ag_chain_sequences.append(row.ag_chain_seq)

            if row.mut_chain == ab_chain[0] and mutate:
                ab_chain_1_sequences.append(
                    mutate_sequence(row.ab_chain1_seq, row.mut_index, mut_aa)
                )
            else:
                ab_chain_1_sequences.append(row.ab_chain1_seq)

            if len(ab_chain) == 2:
                names_ab2.append(f"{row.complex}_{ab_chain[1]}")
                if row.mut_chain == ab_chain[1] and mutate:
                    ab_chain_2_sequences.append(
                        mutate_sequence(row.ab_chain2_seq, row.mut_index, mut_aa)
                    )
                else:
                    ab_chain_2_sequences.append(row.ab_chain2_seq)
        names = names_ag + names_ab1 + names_ab2
        sequences = ag_chain_sequences + ab_chain_1_sequences + ab_chain_2_sequences
        return names, sequences

    elif sequence_type == "joined_chains":
        names, sequences = [], []
        for row in df.itertuples():
            pdb_id, ab_chain, ag_chain, mutation = row.complex.split("_")
            mut_aa = mutation[-1]
            names.append(f"{row.complex}")
            sequence = row.ag_chain_seq
            sequence += "G" * GLYCINE_LINKER_LENGTH
            sequence += row.ab_chain1_seq
            if len(ab_chain) == 2:
                sequence += "G" * GLYCINE_LINKER_LENGTH
                sequence += row.ab_chain2_seq
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
            if mutate:
                sequences.append(mutate_sequence(sequence, mut_index, mut_aa))
            else:
                sequences.append(sequence)
        return names, sequences


def parse_fasta_sequences(fasta_file: Path) -> tuple[list[str], list[str]]:
    """
    Parse a FASTA file and extract sequence descriptions and sequences.

    :param fasta_file: Path to the FASTA file or a file-like object.
    :return: Tuple of (list of descriptions, list of sequences).
    """

    sequence_descriptions = []
    sequence_strings = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract the description and sequence string for each record
        sequence_descriptions.append(record.description)
        sequence_strings.append(str(record.seq))

    return sequence_descriptions, sequence_strings


def generate_esm_embeddings(
    model,
    last_layer: int,
    batch_converter,
    sequences: list[tuple[str, str]],
    sequence_type: Literal["separate_chains", "joined_chains"],
    average_embeddings: bool = False,
    device: str = "cuda:0",
    batch_size: int = 32,
) -> dict[str, torch.FloatTensor]:
    """Generate embeddings using an ESM2 model from https://github.com/facebookresearch/esm.

    :param model: A pretrained ESM2 model.
    :param last_layer: Last layer of the ESM2 model, which will be used to extract embeddings.
    :param batch_converter: A BatchConverter for preparing protein sequences as input.
    :param sequences: A list of tuples of (name, sequence) for the proteins.
    :param average_embeddings: Whether to average the residue embeddings for each protein.
    :param device: The device to use (e.g., "cpu" or "cuda") for the model.
    :param batch_size: The number of sequences to process at once.
    :return: A dictionary mapping protein name to per-residue ESM2 embedding.
    """
    # Move model to device
    model = model.to(device)

    # Compute all embeddings
    start = time()
    name_to_embedding = {}

    with torch.no_grad():
        # Iterate over batches of sequences
        for i in trange(0, len(sequences), batch_size):
            # Get batch of sequences
            batch_sequences = sequences[i : i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)
            batch_tokens = batch_tokens.to(device)

            # Compute embeddings
            results = model(
                batch_tokens, repr_layers=[last_layer], return_contacts=False
            )

            # Get per-residue embeddings
            batch_embeddings = results["representations"][last_layer].cpu()

            # Map sequence name to embedding
            for (name, sequence), embedding in zip(batch_sequences, batch_embeddings):

                # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1
                embedding = embedding[1 : len(sequence) + 1]

                # Optionally average embeddings for each sequence
                if average_embeddings:
                    embedding = embedding.mean(dim=0)

                if sequence_type == "separate_chains":
                    chain_id = name.split("_")[-1]
                    complex_id = name[:-2]
                    name_to_embedding.setdefault(complex_id, {})[chain_id] = embedding
                elif sequence_type == "joined_chains":
                    name_to_embedding[name] = embedding
                else:
                    raise ValueError(f"Invalid sequence type: {sequence_type}")

    print(f"Time = {time() - start} seconds for {len(sequences):,} sequences")
    return name_to_embedding


def generate_embeddings(
    save_path: Path,
    sequences_path: Path,
    sequence_type: Literal["separate_chains", "joined_chains"],
    mutate: bool = False,
    last_layer: int = 33,
    esm_model: str = "esm2_t33_650M_UR50D",
    average_embeddings: bool = False,
    device: str = "cuda:0",
    batch_size: int = 32,
    hub_dir: str | None = None,
) -> None:
    """Generate antigen/antibody embeddings using the ESM2 model from https://github.com/facebookresearch/esm.

    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :param last_layer: Last layer of the ESM2 model, which will be used to extract embeddings.
    :param average_embeddings: Whether to average the residue embeddings for each protein.
    :param save_path: Path to PT file where a dictionary mapping protein name to embeddings will be saved.
    :param sequences_path: Path to a file containing antibody sequences.
    :param device: The device to use (e.g., "cpu" or "cuda") for the model.
    :param batch_size: The number of sequences to process at once.
    :param hub_dir: Path to directory where torch hub models are saved.
    """
    # Load sequences
    df = pd.read_csv(sequences_path)
    names, sequences = parse_and_mutate_sequences(df, sequence_type, mutate)

    # Print stats
    print(f"Number of sequences = {len(sequences):,}")

    # Load ESM-2 model
    model, alphabet, batch_converter = load_esm_model(
        hub_dir=hub_dir, esm_model=esm_model
    )

    # Generate embeddings
    sequence_representations = generate_esm_embeddings(
        model=model,
        last_layer=last_layer,
        batch_converter=batch_converter,
        sequences=list(zip(names, sequences)),
        sequence_type=sequence_type,
        average_embeddings=average_embeddings,
        device=device,
        batch_size=batch_size,
    )

    # Save embeddings
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequence_representations, save_path)


if __name__ == "__main__":
    tapify(generate_embeddings)
