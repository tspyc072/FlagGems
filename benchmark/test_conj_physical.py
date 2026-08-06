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


@pytest.mark.conj_physical
def test_conj_physical():
    bench = base.UnaryPointwiseBenchmark(
        op_name="conj_physical",
        torch_op=torch.conj_physical,
        dtypes=consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conj_physical)
    bench.run()
