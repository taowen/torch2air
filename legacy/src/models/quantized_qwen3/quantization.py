from __future__ import annotations


Q8_TENSOR_NAMES: tuple[str, ...] = ()


def qwen3_q4_k_m_uses_q6_layer(layer_idx: int, num_hidden_layers: int) -> bool:
    if layer_idx < 0 or layer_idx >= num_hidden_layers:
        raise ValueError(f"layer_idx {layer_idx} outside [0, {num_hidden_layers})")
    return False


def qwen3_q4_k_m_q6_tensor_names(num_hidden_layers: int) -> tuple[str, ...]:
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
    return ("lm_head.weight",)
