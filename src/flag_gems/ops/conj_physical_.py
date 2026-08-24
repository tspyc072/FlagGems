import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conj_physical_")
    or runtime.get_tuned_config("conj_physical"),
    key=["n_elements"],
)
@triton.jit
def conj_physical__kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    base = offsets * 2
    real = tl.load(in_ptr + base, mask=mask)
    imag = tl.load(in_ptr + base + 1, mask=mask)

    tl.store(out_ptr + base, real, mask=mask)
    tl.store(out_ptr + base + 1, -imag, mask=mask)


def conj_physical_(input: torch.Tensor) -> torch.Tensor:
    """
    In-place physical conjugate.
    For real tensors, returns the input directly.
    For complex tensors, modifies the storage in-place.
    """
    logger.debug("GEMS CONJ_PHYSICAL_")
    if not input.is_complex():
        return input

    if not input.is_contiguous():
        raise RuntimeError(
            "conj_physical_ only supports contiguous tensors. "
            "Please call .contiguous() before this operation."
        )

    n_elements = input.numel()
    real_ptr = torch.view_as_real(input)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    conj_physical__kernel[grid](real_ptr, real_ptr, n_elements)

    return input
