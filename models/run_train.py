# Train models

from models.model_base_class import ModelJointChain_MLP, Residue_Transformer
from models.dataset import create_dataset_and_dataloader
import torch
import torch.nn as nn
from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader
import wandb
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from tap import tapify


def compute_param_norm(model: ModelJointChain_MLP):
    return sum(p.norm().item() for p in model.parameters())


def compute_grad_norm(model: ModelJointChain_MLP):
    return sum(
        p.grad.data.norm().item() for p in model.parameters() if p.grad is not None
    )


def compute_param_count(model: ModelJointChain_MLP):
    return sum(p.numel() for p in model.parameters())


def log_metrics(true_y: torch.Tensor, pred_y: torch.Tensor, split: str):
    r2 = r2_score(true_y, pred_y)
    mae = mean_absolute_error(true_y, pred_y)
    mse = mean_squared_error(true_y, pred_y)
    pearson = pearsonr(true_y, pred_y)[0]
    spearman = spearmanr(true_y, pred_y)[0]
    wandb.log({f"{split} R2": r2})
    wandb.log({f"{split} MAE": mae})
    wandb.log({f"{split} MSE": mse})
    wandb.log({f"{split} Pearson": pearson})
    wandb.log({f"{split} Spearman": spearman})


def eval_model(
    model: ModelJointChain_MLP | Residue_Transformer,
    eval_dataloader: DataLoader,
    split: str,
    device: str,
    plot_scatter: bool = False,
):
    model.eval()
    pred_y = []
    true_y = []
    with torch.no_grad():
        for batch in eval_dataloader:
            x, y = batch
            x = x.to(device)
            pred_y.append(model(x).cpu())
            true_y.append(y)
    pred_y = torch.cat(pred_y)
    true_y = torch.cat(true_y)
    if plot_scatter:
        data = [[x, y] for (x, y) in zip(true_y, pred_y)]
        table = wandb.Table(data=data, columns=["true_ddg", "pred_ddg"])
        wandb.log(
            {
                f"ddg_scatter_{split}": wandb.plot.scatter(
                    table,
                    "true_ddg",
                    "pred_ddg",
                    title=f"Predicted vs True ddG ({split})",
                )
            }
        )

    log_metrics(true_y, pred_y, split)


def train_model(
    model: ModelJointChain_MLP | Residue_Transformer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    weight_decay: float = 0,
):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            x, y, padding_mask = batch
            x = x.to(device)
            y = y.to(device)
            padding_mask = padding_mask.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, padding_mask), y)
            wandb.log({"Train Loss vs Batch": loss.item()})
            loss.backward()
            optimizer.step()
            wandb.log(
                {
                    "Param Norm": compute_param_norm(model),
                    "Grad Norm": compute_grad_norm(model),
                }
            )
        # evaluate on train set
        eval_model(model, train_dataloader, "train", device)
        # evaluate on val set
        eval_model(model, val_dataloader, "val", device)
    return model


def run_train(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    mutant_embedding_path: Path,
    wt_embedding_path: Path,
    model_type: Literal["MLP", "Residue_Transformer"],
    sequence_type: Literal["separate_chains", "joined_chains"],
    hidden_dim: int,
    num_layers: int,
    dropout_rate: float,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device: str,
    weight_decay: float = 0,
    clip_outliers: bool = False,
    min_ddg: float = -10,
    max_ddg: float = 10,
):
    start_time = time.time()
    # run = wandb.init(
    #     # Set the wandb project where this run will be logged.
    #     project="ab-ag-ddg",
    #     name=f"MLP_diff_{sequence_type}_{hidden_dim}_{num_layers}_{learning_rate}_{dropout_rate}_{weight_decay}",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "learning_rate": learning_rate,
    #         "architecture": "MLP",
    #         "dataset": "Flex_ddG",
    #         "epochs": num_epochs,
    #         "hidden_dim": hidden_dim,
    #         "num_layers": num_layers,
    #         "dropout_rate": dropout_rate,
    #         "weight_decay": weight_decay,
    #         "sequence_type": sequence_type,
    #         "batch_size": batch_size,
    #         "device": device,
    #         "train_path": train_path,
    #         "val_path": val_path,
    #         "test_path": test_path,
    #         "mutant_embedding_path": mutant_embedding_path,
    #         "wt_embedding_path": wt_embedding_path,
    #         "clip_outliers": clip_outliers,
    #         "min_ddg": min_ddg,
    #         "max_ddg": max_ddg,
    #     },
    # )

    train_dataset, train_dataloader = create_dataset_and_dataloader(
        train_path,
        mutant_embedding_path,
        wt_embedding_path,
        sequence_type,
        batch_size,
        shuffle=True,
        clip_outliers=clip_outliers,
        min_ddg=min_ddg,
        max_ddg=max_ddg,
    )
    breakpoint()
    val_dataset, val_dataloader = create_dataset_and_dataloader(
        val_path,
        mutant_embedding_path,
        wt_embedding_path,
        sequence_type,
        batch_size,
        shuffle=False,
        clip_outliers=clip_outliers,
        min_ddg=min_ddg,
        max_ddg=max_ddg,
    )
    test_dataset, test_dataloader = create_dataset_and_dataloader(
        test_path,
        mutant_embedding_path,
        wt_embedding_path,
        sequence_type,
        batch_size,
        shuffle=False,
        clip_outliers=clip_outliers,
        min_ddg=min_ddg,
        max_ddg=max_ddg,
    )
    if model_type == "MLP":
        model = ModelJointChain_MLP(
            input_dim=train_dataset.embeddings.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
    elif model_type == "Residue_Transformer":
        model = Residue_Transformer(
            input_dim=train_dataset.embeddings.shape[2],
            hidden_dim=hidden_dim,
        )
    wandb.log({"Param Count": compute_param_count(model)})
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs,
        learning_rate,
        device,
        weight_decay,
    )
    eval_model(model, test_dataloader, "test", device, plot_scatter=True)
    eval_model(model, val_dataloader, "val", device, plot_scatter=True)
    eval_model(model, train_dataloader, "train", device, plot_scatter=True)
    run.log({f"Time to train {num_epochs} epochs": time.time() - start_time})
    run.finish()


def run_train_single_model(
    hidden_dim: int,
    learning_rate: float,
    model_type: Literal["MLP", "Residue_Transformer"],
    num_layers: int = 1,
    dropout_rate: float = 0,
    weight_decay: float = 0,
):

    train_path = Path(
        "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_train.csv"
    )
    val_path = Path(
        "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_val.csv"
    )
    test_path = Path(
        "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_test.csv"
    )
    mutant_embedding_path = Path(
        "/home/dkannan/orcd/scratch/ab-ag-ddg/embeddings_joined_chains_mutants_res.pt"
    )
    wt_embedding_path = Path(
        "/home/dkannan/orcd/scratch/ab-ag-ddg/embeddings_joined_chains_wt_res.pt"
    )
    sequence_type = "joined_chains"
    batch_size = 128
    num_epochs = 300
    device = "cuda"
    run_train(
        train_path,
        val_path,
        test_path,
        mutant_embedding_path,
        wt_embedding_path,
        "Residue_Transformer",
        sequence_type,
        hidden_dim,
        num_layers,
        dropout_rate,
        batch_size,
        num_epochs,
        learning_rate,
        device,
        weight_decay,
    )


def hyperparameter_search():
    """
    Hyperparameter search for the MLP model.
    """
    train_path = Path(
        "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_train.csv"
    )
    val_path = Path(
        "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_val.csv"
    )
    test_path = Path(
        "ddg_synthetic/Flex_ddG/cdr_seqid_cutoffs/Synthetic_FlexddG_ddG_20829-cutoff_70_test.csv"
    )
    mutant_embedding_path = Path(
        "ddg_synthetic/Flex_ddG/embeddings/embeddings_joined_chains_mutants.pt"
    )
    wt_embedding_path = Path(
        "ddg_synthetic/Flex_ddG/embeddings/embeddings_joined_chains_wt.pt"
    )
    sequence_type = "joined_chains"
    batch_size = 128
    dropout_rate = 0.2
    weight_decay = 0
    num_epochs = 300
    device = "cuda"
    for hidden_dim in [32, 128, 256]:
        for num_layers in [2, 3, 4]:
            for learning_rate in [0.0001, 0.001, 0.01]:
                run_train(
                    train_path,
                    val_path,
                    test_path,
                    mutant_embedding_path,
                    wt_embedding_path,
                    sequence_type,
                    hidden_dim,
                    num_layers,
                    dropout_rate,
                    batch_size,
                    num_epochs,
                    learning_rate,
                    device,
                    weight_decay,
                )


if __name__ == "__main__":
    tapify(run_train_single_model)
