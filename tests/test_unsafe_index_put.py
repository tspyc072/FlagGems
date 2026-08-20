import pytest
import torch

import flag_gems
from . import accuracy_utils as utils

@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("shape", [(64, 64), (128, 256), (512, 512)])
@pytest.mark.parametrize("k", [1, 8, 32])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_index_put_dim0(shape, k, dtype):
    device = flag_gems.device
    self_ = torch.randn(shape, dtype=dtype, device=device)
    # unique indices only: with duplicate indices and accumulate=False,
    # the result is undefined per PyTorch docs
    idx = torch.randperm(shape[0], device=device)[:k]
    v = torch.randn((k,) + shape[1:], dtype=dtype, device=device)

    ref_out = torch.ops.aten._unsafe_index_put(
        utils.to_reference(self_), [utils.to_reference(idx)],
        utils.to_reference(v), False,
    )
    snapshot = self_.clone()
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_index_put(self_, [idx], v, False)

    utils.gems_assert_equal(res_out, ref_out)
    # functional semantics: the input must NOT be mutated,
    # and the result must be a new tensor
    utils.gems_assert_equal(self_, utils.to_reference(snapshot, upcast=False))
    assert res_out.data_ptr() != self_.data_ptr()

@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("shape", [(32, 48, 40)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_index_put_nonadjacent(shape, dtype):
    """Non-adjacent indexed dims, equivalent to x[idx0, :, idx2] = v."""
    device = flag_gems.device
    k = 10
    idx0 = torch.randperm(shape[0], device=device)[:k]
    idx2 = torch.randperm(shape[2], device=device)[:k]
    self_ = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(k, shape[1], dtype=dtype, device=device)

    ref_out = torch.ops.aten._unsafe_index_put(
        utils.to_reference(self_),
        [utils.to_reference(idx0), None, utils.to_reference(idx2)],
        utils.to_reference(v), False,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_index_put(self_, [idx0, None, idx2], v, False)

    utils.gems_assert_equal(res_out, ref_out)

@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("shape", [(64, 96, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_index_put_multi_index(shape, dtype):
    """Adjacent multi-dim indices, equivalent to x[idx0, idx1] = v."""
    device = flag_gems.device
    k = 16
    idx0 = torch.randperm(shape[0], device=device)[:k]
    idx1 = torch.randperm(shape[1], device=device)[:k]
    self_ = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(k, shape[2], dtype=dtype, device=device)

    ref_out = torch.ops.aten._unsafe_index_put(
        utils.to_reference(self_),
        [utils.to_reference(idx0), utils.to_reference(idx1)],
        utils.to_reference(v), False,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_index_put(self_, [idx0, idx1], v, False)

    utils.gems_assert_equal(res_out, ref_out)

@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_index_put_accumulate(shape, dtype):
    """accumulate=True: with unique indices the result is order-independent,
    so a tolerance-based comparison is safe."""
    device = flag_gems.device
    k = 64
    idx = torch.randperm(shape[0], device=device)[:k]
    self_ = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(k, shape[1], dtype=dtype, device=device)

    ref_out = torch.ops.aten._unsafe_index_put(
        utils.to_reference(self_), [utils.to_reference(idx)],
        utils.to_reference(v), True,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_index_put(self_, [idx], v, True)

    utils.gems_assert_close(res_out, ref_out, dtype)

@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("shape", [(64, 64)])
def test_unsafe_index_put_broadcast_values(shape):
    """Values broadcasting: (k, 1) broadcast to (k, C)."""
    device = flag_gems.device
    dtype = torch.float32
    k = 16
    idx = torch.randperm(shape[0], device=device)[:k]
    self_ = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(k, 1, dtype=dtype, device=device)

    ref_out = torch.ops.aten._unsafe_index_put(
        utils.to_reference(self_), [utils.to_reference(idx)],
        utils.to_reference(v), False,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_index_put(self_, [idx], v, False)

    utils.gems_assert_equal(res_out, ref_out)

@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("shape", [(64, 64)])
def test_unsafe_index_put_negative_index(shape):
    """Negative indices: native _unsafe_index_put wraps them around
    (-1 refers to the last row). This test probes whether the existing
    codegen kernel matches that behavior."""
    device = flag_gems.device
    dtype = torch.float32
    k = 8
    idx = torch.randperm(shape[0], device=device)[:k] - shape[0]  # all negative
    self_ = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(k, shape[1], dtype=dtype, device=device)

    ref_out = torch.ops.aten._unsafe_index_put(
        utils.to_reference(self_), [utils.to_reference(idx)],
        utils.to_reference(v), False,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_index_put(self_, [idx], v, False)

    utils.gems_assert_equal(res_out, ref_out)
