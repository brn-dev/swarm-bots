"""Optional CUDA kernel with global bindings for Triton compilers without closure support."""

import triton
import triton.language as tl


@triton.jit
def masked_normal_kernel(
    noise: tl.tensor, mask: tl.tensor, seed: tl.tensor, width: tl.constexpr,
    shape: tl.constexpr, strides: tl.constexpr, contiguous: tl.constexpr, block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    if tl.load(mask + row):
        offsets = tl.arange(0, block_size)
        if contiguous:
            addresses = row * width + offsets
        else:
            batch_offset = 0
            batch_index = row
            for axis in tl.static_range(len(shape) - 3, -1, -1):
                batch_offset += (batch_index % shape[axis]) * strides[axis]
                batch_index //= shape[axis]
            addresses = batch_offset + offsets // shape[-1] * strides[-2] + offsets % shape[-1] * strides[-1]
        values = tl.randn(tl.load(seed), row * width + offsets)
        tl.store(noise + addresses, values, offsets < width)
