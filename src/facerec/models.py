"""Modelos que produzem embeddings faciais (ADR-001).

forward(x)  -> embedding L2-normalizado (usado em comparação e nas losses de métrica)
features(x) -> embedding bruto (usado pela cabeça de Cross-Entropy)
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    embedding_dim: int

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.features(x), dim=1)


class SmallCNN(EmbeddingNet):
    """CNN treinada do zero — evolução do `Model(nn.Module)` das aulas: lista de camadas percorrida
    no forward, agora com Conv2d/BatchNorm/MaxPool e uma cabeça que gera o embedding."""

    def __init__(self, embedding_dim: int = 128, image_size: int = 64, in_channels: int = 3,
                 widths=(32, 64, 128)):
        super().__init__()
        blocks, c = [], in_channels
        for w in widths:
            blocks += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2)]
            c = w
        self.layers = nn.ModuleList(blocks)
        feat = image_size // (2 ** len(widths))
        if feat < 1:
            raise ValueError("image_size pequeno demais para o número de blocos")
        self.fc1 = nn.Linear(c * feat * feat, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.embedding_dim = embedding_dim

    def features(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class PretrainedFaceNet(EmbeddingNet):
    """InceptionResnetV1 (facenet-pytorch) pré-treinada em dataset facial.

    finetune: "frozen"  -> nada treina (baseline E01)
              "partial" -> só os últimos blocos (repeat_3, block8, last_linear, last_bn)
              "full"    -> a rede inteira
    """
    PARTIAL = ("repeat_3", "block8", "last_linear", "last_bn")

    def __init__(self, pretrained: "str | None" = "vggface2", finetune: str = "frozen"):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1      # import tardio: só quem usa precisa instalar
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)
        self.embedding_dim = 512
        self.finetune = finetune
        self.set_finetune(finetune)

    def set_finetune(self, mode: str) -> None:
        if mode not in {"frozen", "partial", "full"}:
            raise ValueError(f"finetune inválido: {mode}")
        for name, p in self.backbone.named_parameters():
            p.requires_grad = (mode == "full") or (mode == "partial" and name.startswith(self.PARTIAL))
        self.finetune = mode

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm de partes congeladas não pode atualizar estatísticas
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        return self

    def features(self, x):
        return self.backbone(x)       # a InceptionResnetV1 já devolve L2-normalizado


class ClassifierHead(nn.Module):
    """Cabeça linear usada só no treino com Cross-Entropy (E02)."""

    def __init__(self, embedding_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, f):
        return self.fc(f)


def build_model(model_cfg: dict, image_size: int, load_pretrained: bool = True) -> EmbeddingNet:
    """`load_pretrained=False` constrói só a arquitetura (usado ao carregar um checkpoint nosso,
    para não baixar de novo os pesos originais)."""
    kind = model_cfg["type"]
    if kind == "small_cnn":
        return SmallCNN(model_cfg.get("embedding_dim", 128), image_size)
    if kind == "facenet":
        weights = model_cfg.get("pretrained", "vggface2") if load_pretrained else None
        return PretrainedFaceNet(weights, model_cfg.get("finetune", "frozen"))
    raise ValueError(f"modelo desconhecido: {kind}")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
