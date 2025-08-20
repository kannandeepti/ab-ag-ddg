# Train models

from models.model_base_class import ModelJointChain_MLP
from models.dataset import create_dataset_and_dataloader
import torch
import torch.nn as nn
from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader
import wandb


def eval_model(model: ModelJointChain_MLP, eval_dataloader: DataLoader, device: str):
    loss_fn = nn.MSELoss()
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model(x), y)
            eval_loss += loss.item()
    eval_loss /= len(eval_dataloader)
    return eval_loss


def train_model(
    model: ModelJointChain_MLP,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            train_loss += loss.item()
            wandb.log({"Train Loss vs Batch": loss.item()})
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        # evaluate on val set
        val_loss = eval_model(model, val_dataloader, device)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        wandb.log({"Train Loss vs Epochs": train_loss, "Val Loss vs Epochs": val_loss})
    return model


def run_train(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    embedding_path: Path,
    sequence_type: Literal["separate_chains", "joined_chains"],
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device: str,
):
    run = wandb.init(
        # Set the wandb project where this run will be logged.
        project="ab-ag-ddg",
        name=f"MLP_{sequence_type}_{hidden_dim}_{num_layers}",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": "MLP",
            "dataset": "Flex_ddG",
            "epochs": num_epochs,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "sequence_type": sequence_type,
            "batch_size": batch_size,
            "device": device,
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "embedding_path": embedding_path,
        },
    )

    train_dataset, train_dataloader = create_dataset_and_dataloader(
        train_path, embedding_path, sequence_type, batch_size
    )
    val_dataset, val_dataloader = create_dataset_and_dataloader(
        val_path, embedding_path, sequence_type, batch_size
    )
    test_dataset, test_dataloader = create_dataset_and_dataloader(
        test_path, embedding_path, sequence_type, batch_size
    )

    model = ModelJointChain_MLP(
        input_dim=train_dataset.embeddings.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )
    train_model(
        model, train_dataloader, val_dataloader, num_epochs, learning_rate, device
    )
    test_loss = eval_model(model, test_dataloader, device)
    print(f"Test Loss: {test_loss:.4f}")
    run.log({"Test Loss": test_loss})


if __name__ == "__main__":
    run_train(
        train_path=Path(
            "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_train.csv"
        ),
        val_path=Path(
            "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_val.csv"
        ),
        test_path=Path(
            "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_test.csv"
        ),
        embedding_path=Path(
            "ddg_synthetic/Flex_ddG/embeddings/embeddings_flexddg_joined_chains.pt"
        ),
        sequence_type="joined_chains",
        hidden_dim=128,
        num_layers=2,
        batch_size=128,
        num_epochs=100,
        learning_rate=0.001,
        device="cuda",
    )
