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

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config


def _ascend_conj_ref(input):
    """Ascend reference implementation using small ops combination."""
    if not input.is_complex():
        return input
    return torch.complex(input.real, -input.imag)


def _input_fn(shape, dtype, device):
    if dtype.is_complex:
        if "npu" in str(device) or "ascend" in str(device).lower():
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float16
            real_shape = list(shape) + [2]
            real = torch.randn(real_shape, dtype=real_dtype, device=device)
            inp = torch.view_as_complex(real)
        else:
            inp = torch.randn(shape, dtype=dtype, device=device)
    elif dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    yield (inp,)


class ConjBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        more_shapes_1d = [
            (2**28,),
        ]
        more_shapes_2d = [(10000, 2**i) for i in (0, 8, 16)]
        more_shapes_3d = [(100, 2**i, 100) for i in (0, 8, 16)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


@pytest.mark.conj
def test_conj():
    Config.mode = consts.BenchMode.OPERATOR

    device = torch.randn(1, device=flag_gems.device).device

    if "npu" in str(device) or "ascend" in str(device).lower():
        torch_op = _ascend_conj_ref
    else:
        torch_op = torch.conj

    bench = ConjBenchmark(
        op_name="conj",
        torch_op=torch_op,
        input_fn=_input_fn,
        dtypes=consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES,
    )

    bench.set_gems(flag_gems.conj)
    bench.run()
