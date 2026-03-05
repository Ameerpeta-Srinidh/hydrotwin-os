"""
HydroTwin OS — Plane 2: Multimodal Fusion Transformer

Fuses sensor embeddings, vision embeddings, and vibration embeddings
into a unified representation for anomaly classification.

Architecture:
    Modality Encoders → Cross-Attention Fusion → Classification Head

Output classes: leak, hotspot, vibration_fault, flow_deviation, normal
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


ANOMALY_CLASSES = ["normal", "leak", "hotspot", "vibration_fault", "flow_deviation"]


class ModalityEncoder(nn.Module):
    """Encodes a single modality's raw features into a fixed-size embedding."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion across modalities.

    Each modality attends to all others, allowing the model to learn
    which modality is most informative for each anomaly type.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_modalities, embed_dim]
        Returns:
            Fused representation [batch, num_modalities, embed_dim]
        """
        # Self-attention across modalities
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MultimodalFusionModel(nn.Module):
    """
    Multimodal Fusion Transformer for anomaly classification.

    Fuses three modalities:
        1. Sensor embedding (from statistical features)
        2. Vision embedding (from YOLO/thermal features)
        3. Vibration embedding (from FFT features)

    Outputs:
        - Anomaly class probabilities [normal, leak, hotspot, vibration_fault, flow_deviation]
        - Confidence score
        - Attention weights per modality
    """

    def __init__(
        self,
        sensor_dim: int = 12,
        vision_dim: int = 8,
        vibration_dim: int = 16,
        embed_dim: int = 64,
        num_classes: int = 5,
        num_heads: int = 4,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Modality encoders
        self.sensor_encoder = ModalityEncoder(sensor_dim, embed_dim)
        self.vision_encoder = ModalityEncoder(vision_dim, embed_dim)
        self.vibration_encoder = ModalityEncoder(vibration_dim, embed_dim)

        # Modality type embeddings (learnable)
        self.modality_embeddings = nn.Embedding(3, embed_dim)

        # Cross-attention fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossAttentionFusion(embed_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Modality importance (for interpretability)
        self.modality_gate = nn.Linear(embed_dim * 3, 3)

    def forward(
        self,
        sensor_features: torch.Tensor,
        vision_features: torch.Tensor,
        vibration_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            sensor_features: [batch, sensor_dim]
            vision_features: [batch, vision_dim]
            vibration_features: [batch, vibration_dim]

        Returns:
            Dict with 'logits', 'probs', 'predicted_class', 'modality_weights'
        """
        batch_size = sensor_features.size(0)

        # Encode each modality
        sensor_emb = self.sensor_encoder(sensor_features)      # [B, embed]
        vision_emb = self.vision_encoder(vision_features)      # [B, embed]
        vibration_emb = self.vibration_encoder(vibration_features)  # [B, embed]

        # Add modality type embeddings
        mod_ids = torch.arange(3, device=sensor_features.device)
        mod_embs = self.modality_embeddings(mod_ids)  # [3, embed]
        sensor_emb = sensor_emb + mod_embs[0]
        vision_emb = vision_emb + mod_embs[1]
        vibration_emb = vibration_emb + mod_embs[2]

        # Stack modalities: [B, 3, embed]
        fused = torch.stack([sensor_emb, vision_emb, vibration_emb], dim=1)

        # Cross-attention fusion
        for layer in self.fusion_layers:
            fused = layer(fused)

        # Flatten modalities
        flat = fused.reshape(batch_size, -1)  # [B, 3 * embed]

        # Classification
        logits = self.classifier(flat)
        probs = F.softmax(logits, dim=-1)
        predicted = torch.argmax(probs, dim=-1)

        # Modality importance weights
        gate_logits = self.modality_gate(flat)
        modality_weights = F.softmax(gate_logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "predicted_class": predicted,
            "modality_weights": modality_weights,
            "class_names": ANOMALY_CLASSES,
        }

    def predict(
        self,
        sensor_features: np.ndarray,
        vision_features: np.ndarray,
        vibration_features: np.ndarray,
    ) -> dict[str, Any]:
        """Convenience method for numpy input → dict output."""
        self.eval()
        with torch.no_grad():
            s = torch.tensor(sensor_features, dtype=torch.float32).unsqueeze(0)
            v = torch.tensor(vision_features, dtype=torch.float32).unsqueeze(0)
            vib = torch.tensor(vibration_features, dtype=torch.float32).unsqueeze(0)
            output = self.forward(s, v, vib)

        probs = output["probs"][0].cpu().numpy()
        predicted_idx = int(output["predicted_class"][0].cpu())
        modality_w = output["modality_weights"][0].cpu().numpy()

        return {
            "predicted_class": ANOMALY_CLASSES[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "class_probabilities": {
                name: round(float(p), 4)
                for name, p in zip(ANOMALY_CLASSES, probs)
            },
            "modality_weights": {
                "sensor": round(float(modality_w[0]), 3),
                "vision": round(float(modality_w[1]), 3),
                "vibration": round(float(modality_w[2]), 3),
            },
        }
