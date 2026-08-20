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

import numpy as np
import pytest
import torch

import flag_gems

from . import base, consts

def unsafe_index_put_input_fn(accumulate):
    def inner(shapes, dtype, device):
        input_shape, indices_shape, values_shape = shapes
        inp = torch.randn(
            input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        indices = []
        for i, shape in enumerate(indices_shape):
            index = np.random.choice(
                np.arange(input_shape[i]), size=shape, replace=accumulate
            )
            indices.append(torch.tensor(index, device=flag_gems.device))
        values = torch.randn(
            values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        yield inp, indices, values, accumulate

    return inner

class UnsafeIndexPutAccFalseBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        self.shapes = (
            ((2**28,), ((2**16,),), (2**16,)),
            ((32, 32), ((8,), (8,)), (8,)),
            ((32, 32), ((2, 8),), (32,)),
            ((1024, 1024), ((64,), (64,)), (64,)),
            ((512, 512, 512), ((128,), (128,), (128,)), (128,)),
            ((512, 512, 512), ((2, 128),), (2, 128, 512, 512)),
            # large MoE-dispatch-like shapes
            ((16384, 4096), ((4096,),), (4096, 4096)),
            ((262144, 1024), ((65536,),), (65536, 1024)),
        )
        return None

class UnsafeIndexPutAccTrueBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        self.shapes = (
            ((100,), ((100,),), (100,)),
            ((32, 32), ((32, 32),), (32, 32)),
            ((64, 64, 64), ((64, 64, 64),), (64, 64, 64, 64)),
            ((512, 512), ((512, 512),), (512, 512, 512)),
        )
        return None

@pytest.mark.unsafe_index_put
def test_unsafe_index_put_acc_false():
    bench = UnsafeIndexPutAccFalseBenchmark(
        op_name="_unsafe_index_put",
        torch_op=torch.ops.aten._unsafe_index_put,
        input_fn=unsafe_index_put_input_fn(False),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

@pytest.mark.unsafe_index_put
def test_unsafe_index_put_acc_true():
    bench = UnsafeIndexPutAccTrueBenchmark(
        op_name="_unsafe_index_put",
        torch_op=torch.ops.aten._unsafe_index_put,
        input_fn=unsafe_index_put_input_fn(True),
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
