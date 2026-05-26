from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelStats:
    losses: np.ndarray
    embeddings: np.ndarray
    errors: np.ndarray


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.body(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def make_model(seed: int, hidden_dim: int, device: torch.device) -> SmallMLP:
    torch.manual_seed(seed)
    return SmallMLP(hidden_dim=hidden_dim).to(device)


def as_tensors(x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    weight_decay: float = 0.0,
) -> None:
    if len(y) == 0 or epochs <= 0:
        return
    x_t, y_t = as_tensors(x, y, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()


def compute_losses(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x_t, y_t = as_tensors(x, y, device)
    out = []
    with torch.no_grad():
        for start in range(0, len(y), 4096):
            logits = model(x_t[start : start + 4096])
            loss = F.cross_entropy(logits, y_t[start : start + 4096], reduction="none")
            out.append(loss.detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def eval_error(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    x_t, y_t = as_tensors(x, y, device)
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(y), 4096):
            logits = model(x_t[start : start + 4096])
            pred = logits.argmax(dim=1)
            target = y_t[start : start + 4096]
            correct += int((pred == target).sum().item())
            total += len(target)
    return 1.0 - correct / max(total, 1)


def model_stats(model: SmallMLP, x: np.ndarray, y: np.ndarray, device: torch.device) -> ModelStats:
    model.eval()
    x_t, y_t = as_tensors(x, y, device)
    losses: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(y), 4096):
            logits, emb = model(x_t[start : start + 4096], return_embedding=True)
            target = y_t[start : start + 4096]
            loss = F.cross_entropy(logits, target, reduction="none")
            probs = logits.softmax(dim=1)
            one_hot = F.one_hot(target, num_classes=2).float()
            losses.append(loss.detach().cpu().numpy())
            embeddings.append(emb.detach().cpu().numpy())
            errors.append((probs - one_hot).detach().cpu().numpy())
    return ModelStats(
        losses=np.concatenate(losses) if losses else np.array([], dtype=np.float64),
        embeddings=np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float64),
        errors=np.vstack(errors) if errors else np.empty((0, 2), dtype=np.float64),
    )

