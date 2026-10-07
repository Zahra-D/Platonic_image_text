"""Joint masked-diffusion Transformer for CLEVR text and image tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint

from .lora import (
    IMAGE_PRIVATE_ONLY_ROUTE_ID,
    TEXT_PRIVATE_ONLY_ROUTE_ID,
    lora_modality_context,
    shared_activation_context,
    shared_replacement_context,
)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


def gradient_reverse(inputs: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(inputs, coefficient)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        sdpa_mask = None
        if attention_mask is not None:
            sdpa_mask = attention_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_ffn_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), attention_mask)
        # data2vec generally uses this contextualized FFN write before the
        # block's final residual addition as its teacher target.
        ffn_output = self.mlp(self.norm2(x))
        x = x + ffn_output
        return (x, ffn_output) if return_ffn_output else x


class SharedTokenCrossAttention(nn.Module):
    """Inject condition-shared tokens into a target stream.

    Target residual tokens produce queries.  The exact per-layer shared-LoRA
    representation used by JEPA produces keys and values.  A zero-initialized
    output projection makes adding a new bridge to a pretrained checkpoint an
    identity operation before translation tuning begins.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        mode: str = "projected_cross_attention",
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by translation attention heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        if mode not in {
            "projected_cross_attention", "soft_permutation", "module_replacement"
        }:
            raise ValueError(f"unknown shared translation mode: {mode}")
        self.mode = mode
        self.query_norm = nn.LayerNorm(d_model)
        self.condition_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        if mode == "projected_cross_attention":
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.out_proj.weight)
            self.register_parameter("replacement_gate", None)
        else:
            # The values are the condition shared vectors themselves.  Attention
            # only resamples their token axis from L_condition to L_target.
            self.v_proj = None
            self.out_proj = None
            self.replacement_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        target: torch.Tensor,
        condition_shared: torch.Tensor,
        condition_attention_mask: torch.Tensor | None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, target_length, width = target.shape
        condition_length = condition_shared.size(1)
        q = self.q_proj(self.query_norm(target)).view(
            batch, target_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        condition = self.condition_norm(condition_shared)
        k = self.k_proj(condition).view(
            batch, condition_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        if self.mode == "projected_cross_attention":
            values = self.v_proj(condition)
        else:
            # Do not transform the values: this makes the result a literal soft
            # permutation/resampling of the source shared-token representation.
            values = condition_shared
        v = values.view(batch, condition_length, self.n_heads, self.head_dim).transpose(1, 2)
        attention_mask = (
            None
            if condition_attention_mask is None
            else condition_attention_mask[:, None, None, :]
        )
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = attended.transpose(1, 2).reshape(batch, target_length, width)
        if self.mode == "projected_cross_attention":
            output = self.out_proj(output)
        else:
            # Identity-safe attachment to a pretrained checkpoint.  Once this
            # scalar moves away from zero, Q/K learn the soft token assignment.
            output = output * torch.tanh(self.replacement_gate)
        if not return_attention:
            return output
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / self.head_dim ** 0.5
        if condition_attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        return output, scores.softmax(dim=-1)


class MultimodalMaskedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        modality_adversarial: bool = False,
        modality_discriminator_hidden: int | None = None,
        asymmetric_condition_target: bool = False,
        modality_adversarial_representation_normalization: str = "none",
        use_modality_embeddings: bool = True,
        shared_jepa: bool = False,
        shared_jepa_predictor_hidden_multiplier: int = 2,
        shared_translation_layers: list[int] | tuple[int, ...] | None = None,
        shared_translation_n_heads: int | None = None,
        shared_translation_dropout: float = 0.0,
        shared_translation_mode: str = "projected_cross_attention",
        modulewise_jepa_mode: str = "none",
        modulewise_jepa_layers: list[int] | tuple[int, ...] | None = None,
        modulewise_jepa_modules: list[str] | tuple[str, ...] | None = None,
        data2vec_hidden: bool = False,
        data2vec_layerwise_layers: list[int] | None = None,
        lejepa_projector_dims: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(max_position_embeddings, d_model)
        # This is deliberately optional.  A modality embedding gives every
        # token an explicit text/image label, which can be an undesirable cue
        # when testing whether the shared pathway can discover cross-modal
        # structure without one.
        self.use_modality_embeddings = use_modality_embeddings
        self.modality_embed = nn.Embedding(3, d_model) if use_modality_embeddings else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        # Recompute block activations in backward; set by the trainer.
        self.gradient_checkpointing = False
        self.head = nn.Linear(d_model, vocab_size)
        self.modality_adversarial = modality_adversarial
        self.asymmetric_condition_target = asymmetric_condition_target
        self.shared_jepa = shared_jepa
        translation_layers = tuple(sorted(set(shared_translation_layers or ())))
        if any(layer < 0 or layer >= n_layers for layer in translation_layers):
            raise ValueError("shared translation layers must be valid Transformer indices")
        self.shared_translation_layers = translation_layers
        translation_heads = shared_translation_n_heads or n_heads
        self.shared_translation_bridges = nn.ModuleDict({
            str(layer): SharedTokenCrossAttention(
                d_model, translation_heads, shared_translation_dropout,
                shared_translation_mode,
            )
            for layer in translation_layers
        })
        if shared_jepa_predictor_hidden_multiplier < 1:
            raise ValueError("shared_jepa_predictor_hidden_multiplier must be positive")
        predictor_hidden = d_model * shared_jepa_predictor_hidden_multiplier
        self.shared_jepa_predictors = (
            nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, predictor_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(predictor_hidden, d_model, bias=False),
                )
                for _ in range(n_layers)
            ])
            if shared_jepa and not modulewise_jepa_mode.startswith("data2vec_") else None
        )
        # Faithful data2vec: one prediction head on the student's final hidden
        # state, regressing the EMA teacher's averaged block outputs.  The
        # layerwise variant instead gives every supervised block its own head,
        # so no block predicts another block's code.
        def _data2vec_head() -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, predictor_hidden, bias=False),
                nn.GELU(),
                nn.Linear(predictor_hidden, d_model, bias=False),
            )

        self.data2vec_head = (
            _data2vec_head() if data2vec_hidden and not data2vec_layerwise_layers else None
        )
        self.data2vec_heads = (
            nn.ModuleDict({str(layer): _data2vec_head() for layer in data2vec_layerwise_layers})
            if data2vec_hidden and data2vec_layerwise_layers else None
        )
        # Multi-view LeJEPA projector, as in the reference code:
        # MLP(D, [hidden, hidden, K], norm_layer=BatchNorm1d). SIGReg and the
        # invariance loss act on its output; the backbone embedding is evaluated.
        self.lejepa_projector = None
        if lejepa_projector_dims:
            dims = [d_model, *lejepa_projector_dims]
            layers: list[nn.Module] = []
            for inner, outer in zip(dims[:-2], dims[1:-1]):
                layers += [nn.Linear(inner, outer), nn.BatchNorm1d(outer), nn.ReLU()]
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.lejepa_projector = nn.Sequential(*layers)
        if data2vec_layerwise_layers and any(
            layer < 0 or layer >= n_layers for layer in data2vec_layerwise_layers
        ):
            raise ValueError("data2vec layerwise layers must be valid Transformer indices")
        if modulewise_jepa_mode not in {
            "none", "average", "layerwise", "data2vec_average", "data2vec_no_average"
        }:
            raise ValueError("Unknown modulewise JEPA mode")
        self.modulewise_jepa_mode = modulewise_jepa_mode
        selected_modulewise_layers = tuple(sorted(set(modulewise_jepa_layers or ())))
        if any(layer < 0 or layer >= n_layers for layer in selected_modulewise_layers):
            raise ValueError("modulewise_jepa_layers must be valid Transformer indices")
        if modulewise_jepa_mode != "none" and not selected_modulewise_layers:
            raise ValueError("modulewise JEPA requires at least one selected layer")
        # These predictors consume native adapter updates.  Their widths must
        # match the linear-module output: qkv=3d, out_proj=d, mlp.0=mlp_ratio*d,
        # and mlp.3=d.  They are separate from legacy pooled JEPA predictors.
        allowed_modulewise_modules = ("qkv", "out_proj", "mlp.0", "mlp.3")
        selected_modulewise_modules = tuple(dict.fromkeys(
            modulewise_jepa_modules or ("out_proj", "mlp.3")
        ))
        unknown_modulewise_modules = set(selected_modulewise_modules).difference(
            allowed_modulewise_modules
        )
        if unknown_modulewise_modules:
            raise ValueError(
                f"unknown modulewise JEPA modules: {sorted(unknown_modulewise_modules)}"
            )
        self.modulewise_jepa_modules = selected_modulewise_modules
        modulewise_widths = {
            "qkv": 3 * d_model,
            "out_proj": d_model,
            "mlp.0": mlp_ratio * d_model,
            "mlp.3": d_model,
        }
        predictor_keys = []
        if modulewise_jepa_mode in {"average", "layerwise"}:
            predictor_keys = [
                (f"{module.replace('.', '_')}_l{layer:02d}", modulewise_widths[module])
                for layer in selected_modulewise_layers
                for module in selected_modulewise_modules
            ]
        self.modulewise_jepa_predictors = nn.ModuleDict({
            key: nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, predictor_hidden, bias=False),
                nn.GELU(),
                nn.Linear(predictor_hidden, width, bias=False),
            )
            for key, width in predictor_keys
        })
        if modality_adversarial_representation_normalization not in {"none", "l2"}:
            raise ValueError(
                "modality_adversarial_representation_normalization must be 'none' or 'l2'"
            )
        self.modality_adversarial_representation_normalization = (
            modality_adversarial_representation_normalization
        )
        if modality_adversarial:
            hidden = modality_discriminator_hidden or d_model
            self.modality_discriminator = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2),
            )
        else:
            self.modality_discriminator = None

    def input_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed inputs before dropout, allowing an EMA teacher to share them.

        data2vec shares the feature and positional encoders between student and
        teacher.  The trainer can therefore build clean embeddings with the
        online model and pass them into the EMA Transformer's clean forward.
        """
        values = self.token_embed(input_ids) + self.position_embed(position_ids)
        if self.modality_embed is not None:
            values = values + self.modality_embed(modality_ids)
        return values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        route_ids: torch.Tensor | None = None,
        return_shared: bool = False,
        return_shared_by_layer: bool = False,
        return_shared_tokens_by_layer: bool = False,
        return_shared_native_by_module: bool = False,
        return_shared_private_native_by_module: bool = False,
        return_shared_differentiable_native_by_module: bool = False,
        return_hidden_by_layer: bool = False,
        return_data2vec_by_layer: bool = False,
        input_embeddings: torch.Tensor | None = None,
        private_branch: str = "text",
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            dict[int, torch.Tensor],
            dict[int, torch.Tensor],
        ]
    ):
        if (
            return_shared_by_layer
            or return_shared_tokens_by_layer
            or return_shared_native_by_module
            or return_shared_private_native_by_module
            or return_shared_differentiable_native_by_module
        ) and not return_shared:
            raise ValueError("layerwise shared outputs require return_shared=True")
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device)[None]
        if position_ids.max().item() >= self.max_position_embeddings:
            raise ValueError("position_ids exceed max_position_embeddings")
        if modality_ids is None:
            modality_ids = torch.zeros_like(input_ids)
        if input_embeddings is None:
            x = self.input_embeddings(input_ids, position_ids, modality_ids)
        else:
            expected = (input_ids.size(0), input_ids.size(1), self.token_embed.embedding_dim)
            if input_embeddings.shape != expected:
                raise ValueError(
                    f"input_embeddings has shape {tuple(input_embeddings.shape)}, expected {expected}"
                )
            x = input_embeddings
        x = self.dropout(x)
        with lora_modality_context(route_ids):
            recorder_context = (
                shared_activation_context(attention_mask, self.token_embed.embedding_dim, private_branch)
                if return_shared else nullcontext(None)
            )
            hidden_by_layer: dict[int, torch.Tensor] = {}
            data2vec_by_layer: dict[int, torch.Tensor] = {}
            with recorder_context as recorder:
                for index, block in enumerate(self.blocks):
                    if return_data2vec_by_layer:
                        x, ffn_output = block(
                            x, attention_mask, return_ffn_output=True
                        )
                        data2vec_by_layer[index] = ffn_output
                    elif self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                        x = checkpoint(block, x, attention_mask, use_reentrant=False)
                    else:
                        x = block(x, attention_mask)
                    if return_hidden_by_layer or return_data2vec_by_layer:
                        hidden_by_layer[index] = x
            hidden = self.norm_out(x)
            logits = self.head(hidden)
        if return_data2vec_by_layer:
            # The student predictor consumes residual states, while the clean
            # teacher targets the contextualized FFN writes before their final
            # residual additions.
            return logits, hidden_by_layer, data2vec_by_layer
        if return_hidden_by_layer:
            # data2vec-style objectives need every block's output; the caller
            # gets them alongside the logits and may ignore the rest.
            return logits, hidden_by_layer
        if not return_shared:
            return logits
        shared = recorder.representation if recorder is not None else None
        if shared is None:
            if attention_mask is None:
                shared = hidden.mean(dim=1)
            else:
                weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
                shared = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        if return_shared_differentiable_native_by_module:
            return (
                logits,
                shared,
                recorder.native_by_module if recorder is not None else {},
                recorder.native_private_by_module if recorder is not None else {},
                recorder.native_differentiable_by_module if recorder is not None else {},
            )
        if return_shared_native_by_module:
            return logits, shared, recorder.native_by_module if recorder is not None else {}
        if return_shared_private_native_by_module:
            return (
                logits,
                shared,
                recorder.native_by_module if recorder is not None else {},
                recorder.native_private_by_module if recorder is not None else {},
            )
        if return_shared_tokens_by_layer:
            return (
                logits,
                shared,
                recorder.per_layer_representation if recorder is not None else {},
                recorder.per_layer_token_representation if recorder is not None else {},
            )
        if return_shared_by_layer:
            return logits, shared, recorder.per_layer_representation if recorder is not None else {}
        return logits, shared

    def predict_data2vec(self, hidden: torch.Tensor, layer: int | None = None) -> torch.Tensor:
        if layer is None:
            if self.data2vec_head is None:
                raise RuntimeError("data2vec head is not enabled for this model")
            return self.data2vec_head(hidden)
        if self.data2vec_heads is None or str(layer) not in self.data2vec_heads:
            raise RuntimeError(f"no data2vec prediction head for block {layer}")
        return self.data2vec_heads[str(layer)](hidden)

    def predict_shared_jepa(self, layer: int, representation: torch.Tensor) -> torch.Tensor:
        if self.shared_jepa_predictors is None:
            raise RuntimeError("shared JEPA predictors are disabled")
        return self.shared_jepa_predictors[layer](representation)

    def predict_modulewise_jepa(
        self, branch: str, layer: int, representation: torch.Tensor
    ) -> torch.Tensor:
        key = f"{branch.replace('.', '_')}_l{layer:02d}"
        if key not in self.modulewise_jepa_predictors:
            raise RuntimeError(
                f"No modulewise JEPA predictor for {key}; mode={self.modulewise_jepa_mode}"
            )
        return self.modulewise_jepa_predictors[key](representation)

    @staticmethod
    def _private_only_routes(route_ids: torch.Tensor) -> torch.Tensor:
        """Suppress target-shared LoRA while retaining its private branch."""
        result = route_ids.clone()
        result[route_ids.eq(0)] = TEXT_PRIVATE_ONLY_ROUTE_ID
        result[route_ids.eq(1)] = IMAGE_PRIVATE_ONLY_ROUTE_ID
        return result

    def _embed(self, input_ids, position_ids, modality_ids):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device)[None]
        x = self.token_embed(input_ids) + self.position_embed(position_ids)
        if self.modality_embed is not None:
            x = x + self.modality_embed(modality_ids)
        return self.dropout(x)

    def forward_shared_translation(
        self,
        condition_input_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor,
        condition_position_ids: torch.Tensor,
        condition_modality_ids: torch.Tensor,
        condition_route_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        target_position_ids: torch.Tensor,
        target_modality_ids: torch.Tensor,
        target_route_ids: torch.Tensor,
        *,
        return_bridge_attention: bool = False,
        disable_bridge: bool = False,
        condition_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Translate between separate modality streams through shared L2--L4 K/V.

        The condition stream runs normally.  At configured bridge layers, the
        target stream disables its own shared LoRA, retains its modality-private
        LoRA, and receives a residual cross-attention update whose keys/values
        are the condition's exact JEPA shared-token representation.
        """
        if not self.shared_translation_layers:
            raise RuntimeError("shared translation bridges are not configured")
        if condition_input_ids.size(0) != target_input_ids.size(0):
            raise ValueError("condition and target translation batches must have equal size")
        condition = self._embed(
            condition_input_ids, condition_position_ids, condition_modality_ids
        )
        target = self._embed(target_input_ids, target_position_ids, target_modality_ids)
        target_private_routes = self._private_only_routes(target_route_ids)
        attention_by_layer = {}
        for layer, block in enumerate(self.blocks):
            with lora_modality_context(condition_route_ids):
                recorder_context = (
                    shared_activation_context(condition_attention_mask, self.token_embed.embedding_dim)
                    if layer in self.shared_translation_layers and not disable_bridge
                    else nullcontext(None)
                )
                with recorder_context as recorder:
                    condition = block(condition, condition_attention_mask)
            bridge_active = layer in self.shared_translation_layers and not disable_bridge
            bridge = self.shared_translation_bridges[str(layer)] if bridge_active else None
            if bridge_active and bridge.mode == "module_replacement":
                condition_shared = recorder.per_layer_token_representation.get(layer)
                if condition_shared is None:
                    raise RuntimeError(
                        f"Translation layer {layer} did not record condition shared tokens"
                    )
                # One target-to-source token assignment is shared by qkv,
                # out_proj, mlp.0 and mlp.3.  Each native-width shared delta is
                # then resampled with that same assignment.
                _, attention = bridge(
                    target, condition_shared,
                    condition_attention_mask if condition_kv_mask is None else condition_kv_mask,
                    return_attention=True,
                )
                assignment = attention.mean(dim=1).to(condition_shared.dtype)
                replacements = {
                    name: torch.bmm(assignment, native)
                    * torch.tanh(bridge.replacement_gate)
                    for name, native in recorder.native_by_module.items()
                }
                with lora_modality_context(target_route_ids), shared_replacement_context(replacements):
                    target = block(target, target_attention_mask)
                if return_bridge_attention:
                    attention_by_layer[layer] = attention
                continue
            target_layer_routes = (
                target_private_routes if layer in self.shared_translation_layers else target_route_ids
            )
            with lora_modality_context(target_layer_routes):
                target = block(target, target_attention_mask)
            if bridge_active:
                condition_shared = recorder.per_layer_token_representation.get(layer)
                if condition_shared is None:
                    raise RuntimeError(
                        f"Translation layer {layer} did not record condition shared tokens"
                    )
                bridge_output = bridge(
                    target, condition_shared,
                    condition_attention_mask if condition_kv_mask is None else condition_kv_mask,
                    return_attention=return_bridge_attention,
                )
                if return_bridge_attention:
                    update, attention = bridge_output
                    attention_by_layer[layer] = attention
                else:
                    update = bridge_output
                target = target + update
        logits = self.head(self.norm_out(target))
        return (logits, attention_by_layer) if return_bridge_attention else logits

    def modality_discriminator_input(self, shared: torch.Tensor) -> torch.Tensor:
        """Remove configured nuisance scale before the shared DANN head."""
        if self.modality_adversarial_representation_normalization == "l2":
            return F.normalize(shared, p=2, dim=-1, eps=1e-6)
        return shared

    def modality_logits(self, shared: torch.Tensor, reversal_coefficient: float = 1.0) -> torch.Tensor:
        if self.modality_discriminator is None:
            raise RuntimeError("modality_adversarial is disabled for this model")
        discriminator_input = self.modality_discriminator_input(shared)
        return self.modality_discriminator(
            gradient_reverse(discriminator_input, reversal_coefficient)
        )
