# ==========================================================
# PROJECTOR V3
# Changes:
#   - Input dim: 1024 → 1536  (pre + post + diff)
#   - Input LayerNorm BEFORE MLP (critical for stability)
#   - Num tokens: 32 → 64
#   - Larger hidden dim: 2048 → 2560
#   - Dropout added for regularization
#   - Explicit diff fusion done OUTSIDE this class (in train
#     script) so projector stays clean and testable
# ==========================================================

import torch
import torch.nn as nn

NUM_TOKENS = 64     # more visual tokens = richer grounding
LLM_DIM    = 1536   # Qwen2-1.5B hidden size
CLIP_DIM   = 1536   # 512 pre + 512 post + 512 diff


class MultimodalProjectorV3(nn.Module):

    def __init__(
        self,
        input_dim:  int = CLIP_DIM,
        hidden_dim: int = 2560,
        output_dim: int = LLM_DIM,
        num_tokens: int = NUM_TOKENS,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        # Input norm BEFORE any linear (critical for training stability)
        self.input_norm = nn.LayerNorm(input_dim)

        # Two-layer MLP with mid-point norm
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tokens * output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1536)  — cat([pre_feat, post_feat, post_feat - pre_feat])
        Returns:
            (B, 64, 1536)  — visual token sequence for LLM input
        """
        if x.dim() != 2 or x.shape[1] != CLIP_DIM:
            raise ValueError(
                f"Expected (B, {CLIP_DIM}), got {tuple(x.shape)}. "
                f"Fuse as torch.cat([pre, post, post-pre], dim=1) before calling."
            )
        B = x.shape[0]
        x = self.input_norm(x)                          # normalise before first linear
        x = self.net(x)                                 # (B, 64 * 1536)
        x = x.reshape(B, self.num_tokens, self.output_dim)
        return x