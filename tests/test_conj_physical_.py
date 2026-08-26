import pytest
import torch

import flag_gems


def _conj_physical_ref(x):
    return torch.conj_physical(x).clone()


@pytest.mark.parametrize(
    "shape",
    [
        (256,),
        (1024, 1024),
        (64, 512, 512),
        (2, 3, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_conj_physical__complex(shape, dtype):
    if "npu" in flag_gems.device or "ascend" in flag_gems.device.lower():
        pytest.skip("Ascend NPU does not support complex64/128 randn")

    device = flag_gems.device
    x = torch.randn(shape, dtype=dtype, device=device)
    x_ref = x.clone()

    out = flag_gems.conj_physical_(x)

    assert out is x
    assert torch.allclose(out, _conj_physical_ref(x_ref))


@pytest.mark.parametrize(
    "shape",
    [(128,), (32, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_conj_physical__real(shape, dtype):
    device = flag_gems.device
    x = torch.randn(shape, dtype=dtype, device=device)
    out = flag_gems.conj_physical_(x)

    assert out is x


def test_conj_physical__non_contiguous_error():
    if "npu" in flag_gems.device or "ascend" in flag_gems.device.lower():
        pytest.skip("Ascend NPU does not support complex64/128 randn")

    device = flag_gems.device
    x = torch.randn(4, 4, dtype=torch.complex64, device=device)
    x_non_contig = x.t()

    with pytest.raises(RuntimeError):
        flag_gems.conj_physical_(x_non_contig)
