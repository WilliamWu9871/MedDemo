"""MIL and Token-Guided Multimodal Survival models for Direction 1.

Exp 1.1: GatedABMIL (WSI-only baseline, Ilse et al. 2018 / SurvPath ABMIL pattern).
Exp 1.2: TokenGuidedMultimodalSurv (Paper 1 Fig. 1F: TSR & Clinical Prior Tokens
         as cross-attention semantic anchors over WSI patch embeddings + ViT fusion).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from direction1.losses import hazards_to_survival_and_risk


class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling layer returning pooled representation and weights."""

    def __init__(self, hidden_dim: int, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.attn_v = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool patch tokens ``(1, M, D)`` into ``(1, D)`` and weights ``(M,)``."""
        gated = self.dropout(self.attn_v(h) * self.attn_u(h))
        scores = self.attn_w(gated).squeeze(-1)  # (1, M)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (1, D)
        return pooled, weights.squeeze(0)


class GatedABMIL(nn.Module):
    """Exp 1.1: WSI-only Gated Attention-Based MIL for discrete-time survival."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_bins: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = GatedAttentionPool(hidden_dim=hidden_dim, attn_dim=hidden_dim // 2, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_bins),
        )

    def forward(
        self,
        features: torch.Tensor,
        tsr_vec: torch.Tensor | None = None,
        clinical_vec: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        h = self.patch_proj(features)  # (1, M, D)
        pooled, attn_weights = self.pool(h)
        logits = self.classifier(pooled)
        hazards, survival, risk = hazards_to_survival_and_risk(logits)
        return {
            "logits": logits,
            "hazards": hazards,
            "survival": survival,
            "risk": risk,
            "attn_weights": attn_weights,
        }


class TokenGuidedMultimodalSurv(nn.Module):
    """Exp 1.2: Token-Guided Cross-Attention MIL with TSR Prior & Clinical Tokens.

    Mirrors Paper 1 (Huang et al., Gut 2026, Fig. 1F):
      1. Encodes WSI patches into D-dim visual tokens.
      2. Encodes 8-bin + continuous TSR into a learnable TSR Prior Token ("semantic anchor").
      3. Optionally encodes clinical covariates into a Clinical Token.
      4. Uses Prior Token(s) as Queries against WSI Patch Tokens (Keys/Values) via
         multi-head cross-attention to recalibrate spatial attention toward stromal/prior cues.
      5. Fuses [CLS, WSI_ABMIL, Prior_CrossAttn_Tokens] through a compact ViT encoder block
         and predicts 4-bin discrete survival hazards.
    """

    def __init__(
        self,
        in_dim: int,
        tsr_dim: int = 9,
        clinical_dim: int = 0,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_bins: int = 4,
        dropout: float = 0.25,
        use_tsr: bool = True,
        use_clinical: bool = True,
    ):
        super().__init__()
        self.use_tsr = bool(use_tsr and tsr_dim > 0)
        self.use_clinical = bool(use_clinical and clinical_dim > 0)
        if not self.use_tsr and not self.use_clinical:
            raise ValueError("TokenGuidedMultimodalSurv requires at least one of use_tsr or use_clinical")
        self.hidden_dim = hidden_dim

        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.wsi_pool = GatedAttentionPool(hidden_dim=hidden_dim, attn_dim=hidden_dim // 2, dropout=dropout)

        if self.use_tsr:
            self.tsr_encoder = nn.Sequential(
                nn.Linear(tsr_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.tsr_encoder = None

        if self.use_clinical:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(clinical_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.clinical_encoder = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        n_prior_tokens = int(self.use_tsr) + int(self.use_clinical)
        num_modal_tokens = 2 + n_prior_tokens  # CLS, WSI_pool, [TSR_cross], [Clin_cross]
        self.modality_embed = nn.Parameter(torch.zeros(1, num_modal_tokens, hidden_dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )

        # Residual concat of fused CLS + pooled WSI + enabled prior token(s)
        head_in_dim = hidden_dim * (2 + n_prior_tokens)
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins),
        )

    def forward(
        self,
        features: torch.Tensor,
        tsr_vec: torch.Tensor | None = None,
        clinical_vec: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(0)

        wsi_tokens = self.patch_proj(features)  # (1, M, D)
        wsi_pooled, wsi_attn_weights = self.wsi_pool(wsi_tokens)  # (1, D), (M,)

        prior_queries: list[torch.Tensor] = []
        if self.use_tsr:
            if tsr_vec is None:
                raise ValueError("tsr_vec must be provided when use_tsr=True")
            if tsr_vec.ndim == 1:
                tsr_vec = tsr_vec.unsqueeze(0)
            assert self.tsr_encoder is not None
            tsr_token = self.tsr_encoder(tsr_vec).unsqueeze(1)  # (1, 1, D)
            prior_queries.append(tsr_token)

        if self.use_clinical:
            if clinical_vec is None:
                raise ValueError("clinical_vec must be provided when use_clinical=True")
            if clinical_vec.ndim == 1:
                clinical_vec = clinical_vec.unsqueeze(0)
            assert self.clinical_encoder is not None
            clin_token = self.clinical_encoder(clinical_vec).unsqueeze(1)  # (1, 1, D)
            prior_queries.append(clin_token)

        query_seq = torch.cat(prior_queries, dim=1)  # (1, K, D), K in {1, 2}
        cross_out, cross_weights = self.cross_attn(
            query=query_seq,
            key=wsi_tokens,
            value=wsi_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        cross_tokens = self.cross_norm(query_seq + cross_out)  # (1, K, D)
        # Prior-steered patch attention (query 0: TSR anchor when enabled, else Clinical anchor)
        prior_patch_attn = cross_weights[0, 0, :]  # (M,)

        fusion_seq = torch.cat(
            [self.cls_token, wsi_pooled.unsqueeze(1), cross_tokens],
            dim=1,
        )
        fusion_seq = fusion_seq + self.modality_embed
        fused = self.fusion_transformer(fusion_seq)
        cls_repr = fused[:, 0, :]

        joint_repr = torch.cat([cls_repr, wsi_pooled] + [q.squeeze(1) for q in prior_queries], dim=-1)
        logits = self.classifier(joint_repr)
        hazards, survival, risk = hazards_to_survival_and_risk(logits)
        return {
            "logits": logits,
            "hazards": hazards,
            "survival": survival,
            "risk": risk,
            "attn_weights": prior_patch_attn,
            "wsi_attn_weights": wsi_attn_weights,
        }
