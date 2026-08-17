# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_complex_tensor(shape, dtype, device):
    if "npu" in str(device) or "ascend" in str(device).lower():
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float16
        if len(shape) == 0:
            real = torch.randn(2, dtype=real_dtype, device=device)
            return torch.view_as_complex(real)
        else:
            real_shape = list(shape) + [2]
            real = torch.randn(real_shape, dtype=real_dtype, device=device)
            return torch.view_as_complex(real)
    return torch.randn(shape, dtype=dtype, device=device)


def _assert_complex_close(res, ref):
    ref = ref.to(res.device)
    real_match = torch.all(torch.isclose(res.real, ref.real))
    imag_match = torch.all(torch.isclose(res.imag, ref.imag))
    assert real_match and imag_match, "Complex tensor mismatch"


@pytest.mark.conj
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_complex(shape, dtype):
    device = flag_gems.device
    inp = _make_complex_tensor(shape, dtype, device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.conj(ref_inp)
    res_out = flag_gems.conj(inp)

    _assert_complex_close(res_out, ref_out)
    assert res_out.is_conj(), "Complex conj output must have is_conj() == True"
    if inp.numel() > 0:
        assert (
            inp.data_ptr() == res_out.data_ptr()
        ), "conj must return a view sharing the same underlying storage"

    if "npu" in str(device):
        torch.npu.synchronize()
    del inp, ref_inp, ref_out, res_out
    gc.collect()


@pytest.mark.conj
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conj_real(shape, dtype):
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.conj(ref_inp)
    res_out = flag_gems.conj(inp)

    utils.gems_assert_equal(res_out, ref_out)
    assert (
        res_out.data_ptr() == inp.data_ptr()
    ), "Real conj must return the input tensor itself (identity)"

    if "npu" in str(device):
        torch.npu.synchronize()
    del inp, ref_inp, ref_out, res_out
    gc.collect()
