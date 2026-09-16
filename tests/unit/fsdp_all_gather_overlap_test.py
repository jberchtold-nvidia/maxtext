# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the post-scheduling FSDP all-gather overlap pass."""

from tensorflow.compiler.xla.service import hlo_pb2

from maxtext.utils import fsdp_all_gather_overlap


def _instruction(computation, instruction_id, name, opcode, operands=()):
  instruction = computation.instructions.add(id=instruction_id, name=name, opcode=opcode)
  instruction.operand_ids.extend(operands)
  return instruction


def _module_bytes(*, fsdp_metadata=True):
  module = hlo_pb2.HloModuleProto(name="test")
  inner = module.computations.add(id=2, name="all_gather_body")
  _instruction(inner, 20, "parameter", "parameter")
  _instruction(inner, 21, "all-gather", "all-gather", (20,))

  main = module.computations.add(id=1, name="main")
  _instruction(main, 1, "parameter", "parameter")
  _instruction(main, 8, "independent-gemm", "custom-call", (1,))
  _instruction(main, 9, "weight-projection", "bitcast", (1,))
  start = _instruction(main, 2, "all-gather-start", "async-start", (9,))
  start.called_computation_ids.append(2)
  start.frontend_attributes.map["is_spmd_generated"] = "true" if fsdp_metadata else "false"
  if fsdp_metadata:
    start.metadata.op_name = "jit(step)/TEWrapper_dot_general/custom_partitioning"
  _instruction(main, 3, "more-independent-compute", "fusion", (8,))
  _instruction(main, 4, "all-gather-done", "async-done", (2,))
  _instruction(main, 5, "projection", "get-tuple-element", (4,))
  _instruction(main, 7, "consumer", "custom-call", (5, 3))
  module.schedule.sequences[1].instruction_ids.extend((1, 8, 3, 9, 2, 4, 5, 7))
  module.schedule.sequences[2].instruction_ids.extend((20, 21))
  return module.SerializeToString()


def test_hoists_start_and_weight_projection_across_independent_compute():
  transformed = fsdp_all_gather_overlap.transform_hlo_module(_module_bytes())
  assert transformed is not None
  module = hlo_pb2.HloModuleProto.FromString(transformed)
  assert list(module.schedule.sequences[1].instruction_ids) == [1, 9, 2, 8, 3, 4, 5, 7]
  start = next(i for i in module.computations[1].instructions if i.id == 2)
  assert start.frontend_attributes.map["maxtext_fsdp_overlap"] == "hoist_start_with_input_projections"


def test_ignores_non_fsdp_all_gather():
  assert fsdp_all_gather_overlap.transform_hlo_module(_module_bytes(fsdp_metadata=False)) is None
