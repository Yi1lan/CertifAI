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
    margins: np.ndarray


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        embedding = self.body(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


class MnistFCN(nn.Module):
    def __init__(self, dropout_prob: float = 0.2, num_classes: int = 2) -> None:
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 600)
        self.l2 = nn.Linear(600, 600)
        self.l3 = nn.Linear(600, 600)
        self.l4 = nn.Linear(600, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = F.relu(self.dropout(self.l1(x)))
        x = F.relu(self.dropout(self.l2(x)))
        embedding = F.relu(self.dropout(self.l3(x)))
        logits = self.l4(embedding)
        if return_embedding:
            return logits, embedding
        return logits


class CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.head = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(CifarBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes * CifarBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embedding = out.view(out.shape[0], -1)
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


def infer_model_name(model_name: str, input_shape: tuple[int, ...]) -> str:
    if model_name != "auto":
        return model_name
    if input_shape == (1, 28, 28):
        return "mnist_fcn"
    if input_shape == (3, 32, 32):
        return "cifar_resnet18"
    return "small_mlp"


def make_model(
    seed: int,
    model_name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    hidden_dim: int,
    dropout_prob: float,
    device: torch.device,
) -> nn.Module:
    torch.manual_seed(seed)
    resolved_name = infer_model_name(model_name, input_shape)
    if resolved_name == "small_mlp":
        input_dim = int(np.prod(input_shape))
        model: nn.Module = SmallMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    elif resolved_name == "mnist_fcn":
        model = MnistFCN(dropout_prob=dropout_prob, num_classes=num_classes)
    elif resolved_name == "cifar_resnet18":
        model = CifarResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Valid models: auto, small_mlp, mnist_fcn, cifar_resnet18.")
    return model.to(device)


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
    optimizer_name: str = "adam",
    momentum: float = 0.0,
    nesterov: bool = False,
) -> None:
    if len(y) == 0 or epochs <= 0:
        return
    x_t, y_t = as_tensors(x, y, device)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=bool(nesterov and momentum > 0),
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Valid optimizers: adam, sgd.")
    model.train()
    n = len(y)
    batch_size = n if batch_size <= 0 else batch_size
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()


def compute_losses(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    x_t, y_t = as_tensors(x, y, device)
    out = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            logits = model(x_t[start : start + batch_size])
            loss = F.cross_entropy(logits, y_t[start : start + batch_size], reduction="none")
            out.append(loss.detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def eval_error(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    model.eval()
    x_t, y_t = as_tensors(x, y, device)
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            logits = model(x_t[start : start + batch_size])
            pred = logits.argmax(dim=1)
            target = y_t[start : start + batch_size]
            correct += int((pred == target).sum().item())
            total += len(target)
    return 1.0 - correct / max(total, 1)


def eval_inappropriate_risk(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    losses = compute_losses(model, x, y, device, batch_size)
    return float(np.mean(losses > gamma)) if len(losses) else 0.0


def model_stats(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> ModelStats:
    model.eval()
    x_t, y_t = as_tensors(x, y, device)
    losses: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            logits, emb = model(x_t[start : start + batch_size], return_embedding=True)
            target = y_t[start : start + batch_size]
            loss = F.cross_entropy(logits, target, reduction="none")
            probs = logits.softmax(dim=1)
            if probs.shape[1] >= 2:
                top2 = torch.topk(probs, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
            else:
                margin = probs[:, 0]
            one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
            losses.append(loss.detach().cpu().numpy())
            embeddings.append(emb.detach().cpu().numpy())
            errors.append((probs - one_hot).detach().cpu().numpy())
            margins.append(margin.detach().cpu().numpy())
    return ModelStats(
        losses=np.concatenate(losses) if losses else np.array([], dtype=np.float64),
        embeddings=np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float64),
        errors=np.vstack(errors) if errors else np.empty((0, 0), dtype=np.float64),
        margins=np.concatenate(margins) if margins else np.array([], dtype=np.float64),
    )
