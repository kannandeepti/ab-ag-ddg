from pathlib import Path
from typing import Literal
import torch
import pandas as pd
from torch.nn.modules.module import T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class AntibodyAntigenDataset(Dataset):
    """Custom Dataset for antibody/antigen embeddings."""

    def __init__(
        self,
        ddg_csv_path: Path,
        mutant_embedding_file: Path,
        wt_embedding_file: Path,
        sequence_type: Literal["separate_chains", "joined_chains"],
        clip_outliers: bool = False,
        min_ddg: float = -10,
        max_ddg: float = 10,
    ):
        """
        Args:
            embeddings: List of embedding tensors, each of shape (batch_size, embedding_dim)
            labels: Optional labels tensor of shape (batch_size,)
        """
        df = pd.read_csv(ddg_csv_path)
        name_to_mutant_embeddings = torch.load(mutant_embedding_file)
        name_to_wt_embeddings = torch.load(wt_embedding_file)

        embeddings = []
        labels = []

        for row in df.itertuples():
            name = row.complex
            complex_id, mutation = name.split("_")
            ab_chain = row.ab_chain
            ag_chain = row.ag_chain
            full_name = f"{complex_id}_{ab_chain}_{ag_chain}_{mutation}"
            ddg = row.labels
            try:
                if sequence_type == "separate_chains":
                    ag_chain_embedding = (
                        name_to_mutant_embeddings[full_name][ag_chain]
                        - name_to_wt_embeddings[full_name][ag_chain]
                    )
                    ab_chain1_embedding = (
                        name_to_mutant_embeddings[full_name][ab_chain[0]]
                        - name_to_wt_embeddings[full_name][ab_chain[0]]
                    )
                    if len(ab_chain) > 1:
                        ab_chain2_embedding = (
                            name_to_mutant_embeddings[full_name][ab_chain[1]]
                            - name_to_wt_embeddings[full_name][ab_chain[1]]
                        )
                    else:
                        ab_chain2_embedding = torch.zeros_like(ab_chain1_embedding)

                    embedding = torch.cat(
                        [ag_chain_embedding, ab_chain1_embedding, ab_chain2_embedding]
                    )
                elif sequence_type == "joined_chains":
                    embedding = (
                        name_to_mutant_embeddings[full_name]
                        - name_to_wt_embeddings[full_name]
                    )
                else:
                    raise ValueError(f"Invalid sequence type: {sequence_type}")
            except Exception as e:
                breakpoint()

            embeddings.append(embedding)
            labels.append(ddg)

        self.embeddings = embeddings
        self.labels = labels
        # self.embeddings = torch.stack(embeddings)
        # self.labels = torch.tensor(labels)
        if clip_outliers:
            self.labels = self.labels.clamp(min_ddg, max_ddg)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Return tuple of embeddings for this index
        return self.embeddings[idx], self.labels[idx]


def collate_batch(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad embeddings to the same length and return padding mask.
    Args:
        batch: List of tuples of (embeddings, labels)
    Returns:
        Tuple of (padded_embeddings, labels, padding_mask)
    """
    embeddings, labels = zip(*batch)
    lengths = torch.tensor([len(e) for e in embeddings])
    max_length = lengths.max()
    arange = torch.arange(max_length).unsqueeze(0)  # (1, max_length)
    padding_mask = arange.expand(len(embeddings), max_length) >= lengths.unsqueeze(
        1
    )  # (B, max_length)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    return padded_embeddings, torch.stack(labels), padding_mask


def create_dataset_and_dataloader(
    ddg_csv_path: Path,  # path to csv file
    mutant_embedding_file: Path,
    wt_embedding_file: Path,
    sequence_type: Literal["separate_chains", "joined_chains"],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    clip_outliers: bool = False,
    min_ddg: float = -10,
    max_ddg: float = 10,
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
    dataset = AntibodyAntigenDataset(
        ddg_csv_path,
        mutant_embedding_file,
        wt_embedding_file,
        sequence_type,
        clip_outliers,
        min_ddg,
        max_ddg,
    )
    breakpoint()
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    breakpoint()
    return dataset, dataloader


if __name__ == "__main__":
    dataset = create_dataset_and_dataloader(
        Path(
            "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_train.csv"
        ),
        Path(
            "/home/dkannan/orcd/scratch/ab-ag-ddg/embeddings_joined_chains_mutants_res.pt"
        ),
        Path("/home/dkannan/orcd/scratch/ab-ag-ddg/embeddings_joined_chains_wt_res.pt"),
        "joined_chains",
        batch_size=128,
    )
