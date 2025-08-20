from pathlib import Path
from typing import Literal
import torch
import pandas as pd
from torch.nn.modules.module import T
from torch.utils.data import Dataset, DataLoader


class AntibodyAntigenDataset(Dataset):
    """Custom Dataset for antibody/antigen embeddings."""

    def __init__(
        self,
        ddg_csv_path: Path,
        embedding_file: Path,
        sequence_type: Literal["separate_chains", "joined_chains"],
    ):
        """
        Args:
            embeddings: List of embedding tensors, each of shape (batch_size, embedding_dim)
            labels: Optional labels tensor of shape (batch_size,)
        """
        df = pd.read_csv(ddg_csv_path)
        name_to_embeddings = torch.load(embedding_file)

        embeddings = []
        labels = []

        for row in df.itertuples():
            name = row.complex
            complex_id, mutation = name.split("_")
            ab_chain = row.ab_chain
            ag_chain = row.ag_chain
            full_name = f"{complex_id}_{ab_chain}_{ag_chain}_{mutation}"
            ddg = row.labels
            if sequence_type == "separate_chains":
                ag_chain_embedding = name_to_embeddings[full_name][ag_chain]
                ab_chain1_embedding = name_to_embeddings[full_name][ab_chain[0]]
                if len(ab_chain) > 1:
                    ab_chain2_embedding = name_to_embeddings[full_name][ab_chain[1]]
                else:
                    ab_chain2_embedding = torch.zeros_like(ab_chain1_embedding)
                embedding = torch.cat(
                    [ag_chain_embedding, ab_chain1_embedding, ab_chain2_embedding]
                )
            elif sequence_type == "joined_chains":
                embedding = name_to_embeddings[full_name]
            else:
                raise ValueError(f"Invalid sequence type: {sequence_type}")

            embeddings.append(embedding)
            labels.append(ddg)

        self.embeddings = torch.stack(embeddings)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Return tuple of embeddings for this index
        return self.embeddings[idx], self.labels[idx]


def create_dataset_and_dataloader(
    ddg_csv_path: Path,  # path to csv file
    embedding_file: Path,
    sequence_type: Literal["separate_chains", "joined_chains"],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> tuple[AntibodyAntigenDataset, DataLoader]:
    """
    Create a PyTorch Dataset and DataLoader from embeddings.

    Args:
        embeddings:
        labels: Optional labels tensor of shape (num_samples,)
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (dataset, dataloader)
    """

    # Create dataset
    dataset = AntibodyAntigenDataset(ddg_csv_path, embedding_file, sequence_type)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataset, dataloader
