import torch


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
