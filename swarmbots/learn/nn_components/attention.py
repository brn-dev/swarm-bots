import torch
from torch import nn
from torch.nn import functional as F


AttentionKeysValues = tuple[torch.Tensor, torch.Tensor]


def project_attention_keys_values(
    attention: nn.MultiheadAttention,
    tokens: torch.Tensor,
) -> AttentionKeysValues:
    width = attention.embed_dim
    key_bias = None if attention.in_proj_bias is None else attention.in_proj_bias[width:2 * width]
    value_bias = None if attention.in_proj_bias is None else attention.in_proj_bias[2 * width:]
    if attention._qkv_same_embed_dim:
        bias = None if attention.in_proj_bias is None else attention.in_proj_bias[width:]
        keys, values = F.linear(tokens, attention.in_proj_weight[width:], bias).chunk(2, dim=-1)
    else:
        keys = F.linear(tokens, attention.k_proj_weight, key_bias)
        values = F.linear(tokens, attention.v_proj_weight, value_bias)
    batch_size, length = tokens.shape[:2]
    head_width = width // attention.num_heads
    return (
        keys.reshape(batch_size, length, attention.num_heads, head_width).transpose(1, 2),
        values.reshape(batch_size, length, attention.num_heads, head_width).transpose(1, 2),
    )


def attention_with_projected_keys_values(
    attention: nn.MultiheadAttention,
    queries: torch.Tensor,
    keys_values: AttentionKeysValues,
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    width = attention.embed_dim
    query_weight = attention.in_proj_weight[:width] if attention._qkv_same_embed_dim else attention.q_proj_weight
    query_bias = None if attention.in_proj_bias is None else attention.in_proj_bias[:width]
    projected_queries = F.linear(queries, query_weight, query_bias)
    batch_size, length = queries.shape[:2]
    projected_queries = projected_queries.reshape(
        batch_size, length, attention.num_heads, width // attention.num_heads,
    ).transpose(1, 2)
    output = F.scaled_dot_product_attention(
        projected_queries,
        *keys_values,
        attn_mask=None if key_mask is None else key_mask[:, None, None, :],
        dropout_p=attention.dropout if attention.training else 0.0,
    )
    output = output.transpose(1, 2).reshape(batch_size, length, width)
    return attention.out_proj(output)


def safe_key_padding_mask(
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if key_padding_mask is None:
        return None, None
    masked_keys = key_padding_mask if key_padding_mask.dtype == torch.bool else key_padding_mask.isneginf()
    empty_rows = masked_keys.all(dim=-1, keepdim=True)
    # Native inference attention returns NaNs for empty rows. Their outputs must
    # be zeroed after attention so the temporary visible keys cannot contribute.
    return key_padding_mask.masked_fill(empty_rows, 0), empty_rows
