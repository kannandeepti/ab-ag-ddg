"""Generate antibody embeddings using the ESM2 model from https://github.com/facebookresearch/esm."""

from time import time
from pathlib import Path
import pandas as pd

import torch
from Bio import SeqIO
from tap import Tap
from tqdm import trange


def parse_csv_with_sequences(csv_path: Path) -> tuple[list[str], list[str]]:
    """
    Parse a CSV file with sequences and extract sequence descriptions and sequences.
    """
    df = pd.read_csv(csv_path)
    # antibody chain 1
    ab_chain1_seqs = df["ab_chain1_seq"].tolist()
    names_ab_chain1 = [f"{row.complex}_{row.ab_chain[0]}" for row in df.itertuples()]
    # antibody chain 2
    df2 = df.dropna(subset="ab_chain2_seq")
    ab_chain2_seqs = df2["ab_chain2_seq"].tolist()
    names_ab_chain2 = [f"{row.complex}_{row.ab_chain[1]}" for row in df2.itertuples()]
    # antigen chain
    ag_chains = df["ag_chain_seq"].tolist()
    names_ag = [f"{row.complex}_{row.ag_chain}" for row in df.itertuples()]

    # combine a single list of names and sequences
    names = names_ab_chain1 + names_ab_chain2 + names_ag
    sequences = ab_chain1_seqs + ab_chain2_seqs + ag_chains
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


def load_esm_model(
    esm_model: str = "esm2_t33_650M_UR50D", hub_dir: str | None = None
) -> tuple:
    """Load an ESM2 model and batch converter.

    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :param hub_dir: Path to directory where torch hub models are saved.
    :return: A tuple of a pretrained ESM2 model and a BatchConverter for preparing protein sequences as input.
    """
    if hub_dir is not None:
        torch.hub.set_dir(hub_dir)
    model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, alphabet, batch_converter


def generate_esm_embeddings(
    model,
    last_layer: int,
    batch_converter,
    sequences: list[tuple[str, str]],
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
                chain_id = name.split("_")[-1]
                complex_id = name[:-2]
                # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1
                embedding = embedding[1 : len(sequence) + 1]

                # Optionally average embeddings for each sequence
                if average_embeddings:
                    embedding = embedding.mean(dim=0)

                name_to_embedding.setdefault(complex_id, {})[chain_id] = embedding

    print(f"Time = {time() - start} seconds for {len(sequences):,} sequences")
    return name_to_embedding


def generate_embeddings(
    esm_model: str,
    last_layer: int,
    save_path: Path,
    average_embeddings: bool = False,
    sequences_path: Path | None = None,
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
    names, sequences = parse_csv_with_sequences(sequences_path)

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
        average_embeddings=average_embeddings,
        device=device,
        batch_size=batch_size,
    )

    # Save embeddings
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequence_representations, save_path)


if __name__ == "__main__":

    class Args(Tap):
        esm_model: str
        """Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm."""
        last_layer: int
        """Last layer of the ESM2 model, which will be used to extract embeddings."""
        save_path: Path
        """Path to PT file where a dictionary mapping protein name to embeddings will be saved."""
        average_embeddings: bool = False
        """Whether to average the residue embeddings for each protein."""
        sequences_path: Path
        """Path to a file containing sequences."""
        device: str = "cuda:0"
        """The device to use (e.g., "cpu" or "cuda") for the model."""
        batch_size: int = 32
        """The number of sequences to process at once."""
        hub_dir: str | None = None
        """Path to directory where torch hub models are saved."""

    generate_embeddings(**Args().parse_args().as_dict())
