"""Models used for the full-data ASTAF-R revision experiments.

The baseline implementations are intentionally labelled "style" models in the
manuscript and response because they are compact, common-code reimplementations
rather than executions of the authors' official repositories.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def row_normalize_adjacency(
    adjacency: np.ndarray | torch.Tensor, add_self_loops: bool = True
) -> torch.Tensor:
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    adjacency = torch.clamp(adjacency, min=0)
    if add_self_loops:
        adjacency = adjacency + torch.eye(
            adjacency.shape[0],
            dtype=adjacency.dtype,
            device=adjacency.device,
        )
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class AdaptiveGraphLearner(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, temperature: float = 0.5):
        super().__init__()
        self.source = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.target = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.temperature = temperature
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.target)

    def forward(self) -> torch.Tensor:
        source = self.source[:, None, :].expand(-1, self.target.shape[0], -1)
        target = self.target[None, :, :].expand(self.source.shape[0], -1, -1)
        scores = F.relu(self.edge_mlp(torch.cat([source, target], dim=-1)).squeeze(-1))
        return torch.softmax(scores / self.temperature, dim=-1)


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        scales: Sequence[int] = (1, 3, 6),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scales = tuple(scales)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=scale, padding=scale // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_dim),
                )
                for scale in self.scales
            ]
        )
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.scales)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes, features = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(batch * nodes, features, steps)
        outputs = []
        for encoder in self.encoders:
            encoded = encoder(flat)
            if encoded.shape[-1] > steps:
                encoded = encoded[..., :steps]
            outputs.append(encoded.mean(dim=-1))
        stacked = torch.stack(outputs, dim=1)
        weights = torch.softmax(self.scale_attention(stacked.mean(dim=1)), dim=-1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused.reshape(batch, nodes, self.hidden_dim), weights.reshape(
            batch, nodes, len(self.scales)
        )


class ResidualDecomposition(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trend = nn.Conv1d(
            1, hidden_dim, kernel_size=7, padding=3, padding_mode="reflect"
        )
        self.seasonal = nn.GRU(1, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, nodes, _ = x.shape
        node_series = x.squeeze(-1).permute(0, 2, 1).reshape(batch * nodes, 1, steps)
        trend = self.trend(node_series).mean(dim=-1).reshape(batch, nodes, self.hidden_dim)
        seasonal_input = node_series.transpose(1, 2)
        seasonal_sequence, _ = self.seasonal(seasonal_input)
        seasonal = seasonal_sequence.mean(dim=1).reshape(batch, nodes, self.hidden_dim)
        raw_mean = node_series.mean(dim=-1).reshape(batch, nodes, 1)
        residual_scalar = raw_mean - trend.mean(dim=-1, keepdim=True) - seasonal.mean(
            dim=-1, keepdim=True
        )
        residual = residual_scalar.expand(-1, -1, self.hidden_dim)
        return trend, seasonal, residual


class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(x, x, x, need_weights=False)
        return output


class EfficientAdaptiveGraphLearner(nn.Module):
    """Low-rank directed graph learner with optional top-k sparsification."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        temperature: float = 0.5,
        top_k: Optional[int] = 20,
    ):
        super().__init__()
        self.source = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.target = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.temperature = float(temperature)
        self.top_k = top_k
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.target)

    def forward(self) -> torch.Tensor:
        scale = math.sqrt(self.source.shape[-1])
        logits = F.relu(self.source @ self.target.transpose(0, 1) / scale)
        logits = logits / max(self.temperature, 1e-4)
        if self.top_k is not None and self.top_k < logits.shape[-1]:
            indices = torch.topk(logits, k=self.top_k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, indices, True)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)


class GraphResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("nm,bmh->bnh", adjacency, self.message(features))
        update = self.dropout(self.output(F.gelu(propagated)))
        return self.norm(features + update)


class ASTAFRPlus(nn.Module):
    """Context-aware ASTAF-R candidate used for validation-controlled development.

    The model retains multi-scale encoding, decomposition, adaptive graph learning,
    and optional attention while adding ordered-history projection, node/calendar
    embeddings, residual graph blocks, and last-observation residual forecasting.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_length: int = 12,
        input_dim: int = 1,
        hidden_dim: int = 32,
        horizon: int = 12,
        num_heads: int = 4,
        dropout: float = 0.1,
        temporal_scales: Sequence[int] = (1, 3, 6),
        graph_temperature: float = 0.5,
        graph_top_k: Optional[int] = 20,
        graph_layers: int = 2,
        attention_mode: str = "residual",
        decomposition_mode: str = "learned",
        decomposition_weights: Sequence[float] = (0.35, 0.25, 0.10),
        use_context: bool = True,
        residual_output: bool = True,
        graph_mode: str = "learned_weighted",
        reference_adjacency: Optional[np.ndarray] = None,
        use_history: bool = True,
        use_node_embedding: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.attention_mode = attention_mode
        self.decomposition_mode = decomposition_mode
        self.use_context = use_context
        self.residual_output = residual_output
        self.graph_mode = graph_mode
        self.use_history = use_history
        self.use_node_embedding = use_node_embedding
        self.requires_starts = use_context

        self.temporal = MultiScaleTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scales=temporal_scales,
            dropout=dropout,
        )
        self.history_projection = nn.Sequential(
            nn.Linear(seq_length, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        if use_context:
            self.time_of_day_embedding = nn.Embedding(288, hidden_dim)
            self.day_of_week_embedding = nn.Embedding(7, hidden_dim)
        else:
            self.time_of_day_embedding = None
            self.day_of_week_embedding = None
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.decomposition = ResidualDecomposition(hidden_dim)
        weight_tensor = torch.as_tensor(decomposition_weights, dtype=torch.float32)
        self.register_buffer("decomposition_weights_fixed", weight_tensor)
        if decomposition_mode == "learned":
            total = weight_tensor.sum()
            probabilities = (weight_tensor / total).clamp_min(1e-6)
            self.decomposition_logits = nn.Parameter(torch.log(probabilities))
            self.register_buffer("decomposition_total", total)
        else:
            self.register_parameter("decomposition_logits", None)

        self.graph_learner = EfficientAdaptiveGraphLearner(
            num_nodes,
            hidden_dim,
            temperature=graph_temperature,
            top_k=graph_top_k,
        )
        self.graph_top_k = graph_top_k
        self.register_buffer("identity", torch.eye(num_nodes))
        if reference_adjacency is None:
            reference_adjacency = np.eye(num_nodes, dtype=np.float32)
        self.register_buffer(
            "reference_adjacency",
            row_normalize_adjacency(reference_adjacency, add_self_loops=False),
        )
        random_scores = torch.rand(num_nodes, num_nodes)
        random_scores.fill_diagonal_(-1)
        shuffled_k = min(graph_top_k or num_nodes, num_nodes)
        shuffled_indices = torch.topk(random_scores, k=shuffled_k, dim=-1).indices
        shuffled = torch.zeros(num_nodes, num_nodes)
        shuffled.scatter_(1, shuffled_indices, 1.0 / shuffled_k)
        self.register_buffer("shuffled_adjacency", shuffled)
        self.graph_blocks = nn.ModuleList(
            [GraphResidualBlock(hidden_dim, dropout) for _ in range(graph_layers)]
        )

        if attention_mode in {"residual", "mandatory"}:
            self.spatial_attention = SpatialAttention(hidden_dim, num_heads, dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)
            self.attention_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.ffn_norm = nn.LayerNorm(hidden_dim)
        else:
            self.spatial_attention = None
            self.attention_norm = None
            self.attention_ffn = None
            self.ffn_norm = None

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def decomposition_weight_values(self) -> torch.Tensor:
        if self.decomposition_logits is not None:
            return self.decomposition_total * torch.softmax(self.decomposition_logits, dim=0)
        return self.decomposition_weights_fixed

    def learned_adjacency(self) -> torch.Tensor:
        return self.graph_learner()

    def effective_adjacency(self) -> torch.Tensor:
        learned = self.learned_adjacency()
        if self.graph_mode == "learned_weighted":
            return learned
        if self.graph_mode == "uniform_learned_topk":
            support = learned > 0
            return support.to(learned.dtype) / support.sum(
                dim=-1, keepdim=True
            ).clamp_min(1)
        if self.graph_mode in {"reference_topk", "correlation_topk"}:
            return self.reference_adjacency
        if self.graph_mode == "shuffled_topk":
            return self.shuffled_adjacency
        if self.graph_mode == "identity":
            return self.identity
        if self.graph_mode == "none":
            return self.identity
        raise ValueError(f"Unknown ASTAFRPlus graph mode: {self.graph_mode}")

    def forward(
        self, x: torch.Tensor, starts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features, _ = self.temporal(x)
        ordered_history = x.squeeze(-1).permute(0, 2, 1)
        if self.use_history:
            features = features + self.history_projection(ordered_history)
        if self.use_node_embedding:
            features = features + self.node_embedding.unsqueeze(0)

        if self.use_context:
            if starts is None:
                starts = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            forecast_start = starts + self.seq_length
            time_of_day = torch.remainder(forecast_start, 288)
            day_of_week = torch.remainder(
                torch.div(forecast_start, 288, rounding_mode="floor"), 7
            )
            context = self.time_of_day_embedding(time_of_day) + self.day_of_week_embedding(
                day_of_week
            )
            features = features + context.unsqueeze(1)

        if self.decomposition_mode != "off":
            trend, seasonal, residual = self.decomposition(x)
            weights = self.decomposition_weight_values()
            features = (
                features
                + weights[0] * trend
                + weights[1] * seasonal
                + weights[2] * residual
            )
        features = self.input_norm(features)

        if self.graph_mode != "none":
            adjacency = self.effective_adjacency()
            for block in self.graph_blocks:
                features = block(features, adjacency)

        if self.attention_mode in {"residual", "mandatory"}:
            attended = self.spatial_attention(features)
            if self.attention_mode == "mandatory":
                features = self.attention_norm(attended)
            else:
                features = self.attention_norm(features + attended)
            features = self.ffn_norm(features + self.attention_ffn(features))

        output = self.decoder(features).permute(0, 2, 1).unsqueeze(-1)
        if self.residual_output:
            output = output + x[:, -1:, :, :]
        return output


class ASTAFR(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        hidden_dim: int = 32,
        horizon: int = 12,
        num_heads: int = 4,
        dropout: float = 0.1,
        gamma: float = 0.15,
        learn_gamma: bool = False,
        graph_mode: str = "blend",
        fixed_adjacency: Optional[np.ndarray] = None,
        decomposition_mode: str = "fixed",
        decomposition_weights: Sequence[float] = (0.35, 0.25, 0.10),
        attention_mode: str = "off",
        temporal_scales: Sequence[int] = (1, 3, 6),
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.graph_mode = graph_mode
        self.decomposition_mode = decomposition_mode
        self.attention_mode = attention_mode

        self.temporal = MultiScaleTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scales=temporal_scales,
            dropout=dropout,
        )
        self.graph_learner = AdaptiveGraphLearner(num_nodes, hidden_dim)
        self.register_buffer("identity", torch.eye(num_nodes))
        if fixed_adjacency is None:
            fixed_adjacency = np.eye(num_nodes, dtype=np.float32)
        self.register_buffer(
            "fixed_adjacency",
            row_normalize_adjacency(fixed_adjacency, add_self_loops=False),
        )

        gamma_tensor = torch.tensor(float(gamma)).clamp(1e-4, 1 - 1e-4)
        if learn_gamma:
            self.gamma_logit = nn.Parameter(torch.logit(gamma_tensor))
        else:
            self.register_buffer("gamma_fixed", gamma_tensor)
            self.register_parameter("gamma_logit", None)

        self.decomposition = ResidualDecomposition(hidden_dim)
        weight_tensor = torch.as_tensor(decomposition_weights, dtype=torch.float32)
        self.register_buffer("decomposition_weights_fixed", weight_tensor)
        if decomposition_mode == "learned":
            total = weight_tensor.sum()
            probabilities = (weight_tensor / total).clamp_min(1e-6)
            self.decomposition_logits = nn.Parameter(torch.log(probabilities))
            self.register_buffer("decomposition_total", total)
        else:
            self.register_parameter("decomposition_logits", None)

        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        if attention_mode in {"gated", "mandatory"}:
            self.spatial_attention = SpatialAttention(hidden_dim, num_heads, dropout)
            self.attention_gate = (
                nn.Parameter(torch.tensor(-2.0)) if attention_mode == "gated" else None
            )
        else:
            self.spatial_attention = None
            self.register_parameter("attention_gate", None)

        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def gamma_value(self) -> torch.Tensor:
        if self.gamma_logit is not None:
            return torch.sigmoid(self.gamma_logit)
        return self.gamma_fixed

    def decomposition_weight_values(self) -> torch.Tensor:
        if self.decomposition_logits is not None:
            return self.decomposition_total * torch.softmax(self.decomposition_logits, dim=0)
        return self.decomposition_weights_fixed

    def learned_adjacency(self) -> torch.Tensor:
        return self.graph_learner()

    def effective_adjacency(self) -> torch.Tensor:
        learned = self.learned_adjacency()
        if self.graph_mode == "identity":
            return self.identity
        if self.graph_mode == "fixed":
            return self.fixed_adjacency
        if self.graph_mode == "fixed_self_loop":
            return row_normalize_adjacency(self.fixed_adjacency + self.identity).to(
                self.identity.device
            )
        if self.graph_mode == "learned":
            return learned
        if self.graph_mode in {"blend", "regularized"}:
            gamma = self.gamma_value()
            return (1 - gamma) * self.identity + gamma * learned
        raise ValueError(f"Unknown graph mode: {self.graph_mode}")

    def regularization_loss(self) -> torch.Tensor:
        if self.graph_mode != "regularized":
            return self.identity.new_zeros(())
        adjacency = self.learned_adjacency()
        entropy = -(adjacency * adjacency.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return 1e-4 * entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.temporal(x)
        if self.decomposition_mode != "off":
            trend, seasonal, residual = self.decomposition(x)
            weights = self.decomposition_weight_values()
            features = (
                features + weights[0] * trend + weights[1] * seasonal + weights[2] * residual
            )

        adjacency = self.effective_adjacency()
        propagated = torch.einsum("nm,bmh->bnh", adjacency, features)
        if self.graph_mode == "learned":
            features = self.graph_norm(propagated)
        else:
            features = self.graph_norm(features + propagated)

        if self.attention_mode in {"gated", "mandatory"}:
            attended = self.spatial_attention(features)
            if self.attention_mode == "gated":
                features = self.attention_norm(
                    features + torch.sigmoid(self.attention_gate) * attended
                )
            else:
                features = self.attention_norm(attended)

        output = self.decoder(self.dropout(features))
        return output.permute(0, 2, 1).unsqueeze(-1)


class ASTAFLegacyStyle(nn.Module):
    """Deterministic full-data reconstruction of the submitted ASTAF backbone.

    The original unused quantile heads are deliberately omitted because they
    do not affect the deterministic forward path and are not validated in the
    revised study.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 32,
        horizon: int = 12,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal = MultiScaleTemporalEncoder(
            input_dim=1,
            hidden_dim=hidden_dim,
            scales=(1, 3, 6),
            dropout=dropout,
        )
        self.decomposition = ResidualDecomposition(hidden_dim)
        self.graph_learner = AdaptiveGraphLearner(num_nodes, hidden_dim)
        self.spatial_attention = SpatialAttention(hidden_dim, num_heads, dropout)
        self.temporal_attention = SpatialAttention(hidden_dim, num_heads, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def learned_adjacency(self) -> torch.Tensor:
        return self.graph_learner()

    def effective_adjacency(self) -> torch.Tensor:
        return self.learned_adjacency()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.temporal(x)
        trend, seasonal, residual = self.decomposition(x)
        features = features + 0.30 * trend + 0.20 * seasonal + 0.10 * residual
        features = torch.einsum("nm,bmh->bnh", self.learned_adjacency(), features)
        spatial = self.spatial_attention(features)
        temporal = self.temporal_attention(spatial)
        fused = spatial + temporal
        return self.decoder(fused).permute(0, 2, 1).unsqueeze(-1)


class Persistence(nn.Module):
    def __init__(self, horizon: int = 12):
        super().__init__()
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:, :, :].expand(-1, self.horizon, -1, -1)


class STIDStyle(nn.Module):
    def __init__(
        self, num_nodes: int, seq_length: int = 12, horizon: int = 12, hidden_dim: int = 64
    ):
        super().__init__()
        self.requires_starts = True
        self.seq_length = seq_length
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        self.time_of_day_embedding = nn.Embedding(288, hidden_dim)
        self.day_of_week_embedding = nn.Embedding(7, hidden_dim)
        self.input_projection = nn.Linear(seq_length, hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self, x: torch.Tensor, starts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        flattened = x.squeeze(-1).permute(0, 2, 1)
        if starts is None:
            starts = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        forecast_start = starts + self.seq_length
        time_of_day = torch.remainder(forecast_start, 288)
        day_of_week = torch.remainder(torch.div(forecast_start, 288, rounding_mode="floor"), 7)
        temporal = self.time_of_day_embedding(time_of_day) + self.day_of_week_embedding(
            day_of_week
        )
        hidden = (
            self.input_projection(flattened)
            + self.node_embedding.unsqueeze(0)
            + temporal.unsqueeze(1)
        )
        return self.mlp(hidden).permute(0, 2, 1).unsqueeze(-1)


class TransformerStyle(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        hidden_dim: int = 32,
        horizon: int = 12,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, features = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, features)
        encoded = self.encoder(self.embedding(flat)).mean(dim=1)
        output = self.output(encoded).reshape(batch, nodes, -1)
        return output.permute(0, 2, 1).unsqueeze(-1)


class STGCNStyle(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adjacency: np.ndarray,
        hidden_dim: int = 32,
        horizon: int = 12,
    ):
        super().__init__()
        self.register_buffer("adjacency", row_normalize_adjacency(adjacency))
        self.temporal_1 = nn.Conv2d(1, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.temporal_2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.output = nn.Linear(hidden_dim, horizon)

    def _graph(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nm,bcmt->bcnt", self.adjacency, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.permute(0, 3, 2, 1)
        hidden = F.gelu(self.temporal_1(hidden))
        hidden = self._graph(hidden)
        hidden = self.norm(F.gelu(self.temporal_2(hidden)))
        hidden = self._graph(hidden).mean(dim=-1).permute(0, 2, 1)
        return self.output(hidden).permute(0, 2, 1).unsqueeze(-1)


class DiffusionGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        diffusion_dim = 3 * (input_dim + hidden_dim)
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(diffusion_dim, 2 * hidden_dim)
        self.candidate = nn.Linear(diffusion_dim, hidden_dim)

    @staticmethod
    def _diffuse(features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        forward = torch.einsum("nm,bmf->bnf", adjacency, features)
        backward = torch.einsum("mn,bmf->bnf", adjacency, features)
        return torch.cat([features, forward, backward], dim=-1)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([inputs, hidden], dim=-1)
        reset, update = torch.sigmoid(
            self.gates(self._diffuse(combined, adjacency))
        ).chunk(2, dim=-1)
        candidate_input = torch.cat([inputs, reset * hidden], dim=-1)
        candidate = torch.tanh(
            self.candidate(self._diffuse(candidate_input, adjacency))
        )
        return update * hidden + (1 - update) * candidate


class DCRNNStyle(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adjacency: np.ndarray,
        input_dim: int = 1,
        hidden_dim: int = 32,
        horizon: int = 12,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.register_buffer("adjacency", row_normalize_adjacency(adjacency))
        self.cell = DiffusionGRUCell(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.new_zeros(x.shape[0], self.num_nodes, self.hidden_dim)
        for step in range(x.shape[1]):
            hidden = self.cell(x[:, step], hidden, self.adjacency)
        return self.output(hidden).permute(0, 2, 1).unsqueeze(-1)


class GraphWaveNetBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.temporal = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, 2),
            dilation=(1, dilation),
            padding=(0, dilation),
        )
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        original_steps = x.shape[-1]
        gates = self.temporal(x)[..., :original_steps]
        candidate, gate = gates.chunk(2, dim=1)
        hidden = torch.tanh(candidate) * torch.sigmoid(gate)
        graph_hidden = torch.einsum("nm,bcmt->bcnt", adjacency, hidden)
        residual = self.residual(graph_hidden)
        return self.norm(x + residual), self.skip(graph_hidden)


class GraphWaveNetStyle(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        adjacency: np.ndarray,
        hidden_dim: int = 32,
        horizon: int = 12,
    ):
        super().__init__()
        self.register_buffer("fixed_adjacency", row_normalize_adjacency(adjacency))
        self.graph_learner = AdaptiveGraphLearner(num_nodes, hidden_dim)
        self.input = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [GraphWaveNetBlock(hidden_dim, dilation) for dilation in (1, 2, 4)]
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, horizon, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x.permute(0, 3, 2, 1))
        adjacency = 0.5 * self.fixed_adjacency + 0.5 * self.graph_learner()
        skip = 0.0
        for block in self.blocks:
            hidden, contribution = block(hidden, adjacency)
            skip = skip + contribution
        output = self.output(skip).mean(dim=-1)
        return output.unsqueeze(-1)


class AGCRNStyle(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        hidden_dim: int = 32,
        horizon: int = 12,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.graph_learner = AdaptiveGraphLearner(num_nodes, hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gates = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.candidate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        adjacency = self.graph_learner()
        hidden = x.new_zeros(batch, self.num_nodes, self.hidden_dim)
        for step in range(x.shape[1]):
            inputs = self.input_projection(x[:, step])
            propagated = torch.einsum("nm,bmh->bnh", adjacency, hidden)
            reset, update = torch.sigmoid(
                self.gates(torch.cat([inputs, propagated], dim=-1))
            ).chunk(2, dim=-1)
            candidate = torch.tanh(
                self.candidate(torch.cat([inputs, reset * propagated], dim=-1))
            )
            hidden = update * hidden + (1 - update) * candidate
        return self.output(hidden).permute(0, 2, 1).unsqueeze(-1)


def build_model(
    name: str,
    num_nodes: int,
    adjacency: np.ndarray,
    seq_length: int = 12,
    horizon: int = 12,
    hidden_dim: int = 32,
    **overrides,
) -> nn.Module:
    normalized_name = name.lower()
    if normalized_name == "persistence":
        return Persistence(horizon)
    if normalized_name == "astafr":
        return ASTAFR(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            horizon=horizon,
            fixed_adjacency=adjacency,
            **overrides,
        )
    if normalized_name == "astafr_plus":
        reference_adjacency = overrides.pop("reference_adjacency", adjacency)
        return ASTAFRPlus(
            num_nodes=num_nodes,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            horizon=horizon,
            reference_adjacency=reference_adjacency,
            **overrides,
        )
    if normalized_name == "astaf":
        return ASTAFLegacyStyle(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            horizon=horizon,
        )
    if normalized_name == "astafr_attention":
        return ASTAFR(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            horizon=horizon,
            fixed_adjacency=adjacency,
            attention_mode="mandatory",
            **overrides,
        )
    if normalized_name == "stid":
        return STIDStyle(num_nodes, seq_length, horizon, hidden_dim)
    if normalized_name == "transformer":
        return TransformerStyle(num_nodes, hidden_dim=hidden_dim, horizon=horizon)
    if normalized_name == "stgcn":
        return STGCNStyle(num_nodes, adjacency, hidden_dim, horizon)
    if normalized_name == "dcrnn":
        return DCRNNStyle(num_nodes, adjacency, hidden_dim=hidden_dim, horizon=horizon)
    if normalized_name == "graphwavenet":
        return GraphWaveNetStyle(num_nodes, adjacency, hidden_dim, horizon)
    if normalized_name == "agcrn":
        return AGCRNStyle(num_nodes, hidden_dim=hidden_dim, horizon=horizon)
    raise ValueError(f"Unknown model: {name}")
