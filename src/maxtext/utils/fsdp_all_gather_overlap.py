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

"""Post-scheduling overlap pass for FSDP weight all-gathers.

XLA's latency-hiding scheduler can leave zero-cost weight projections next to
their FSDP all-gather, even when the projected inputs were available much
earlier.  This pass hoists those projections and the collective start across
independent compute, preserving collective launch order.  A bounded hoist
avoids excessive buffer live ranges while providing grouped GEMMs and
attention work from the previous layer with a chance to cover communication.

The rewrite deliberately runs at POST_SCHEDULER: doing it earlier lets the
latency-hiding scheduler replace the chosen order.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

from maxtext.utils import max_logging


_PASS_NAME = "maxtext-fsdp-all-gather-overlap"
_TRANSPARENT_OPCODES = frozenset(("bitcast", "get-tuple-element"))
_registered = False


def _is_fsdp_all_gather_start(instruction, computations_by_id) -> bool:
  """Recognizes async FSDP gathers without depending on unstable HLO names."""
  if instruction.opcode != "async-start":
    return False
  if instruction.frontend_attributes.map.get("maxtext_collective") == "fsdp_weight_all_gather":
    return True
  if instruction.frontend_attributes.map.get("is_spmd_generated") != "true":
    return False
  # TE weight gathers carry the custom-partitioning source metadata through
  # SPMD. Exclude backward gathers: even short-window backward hoists competed
  # with EP/gradient collectives and regressed the profiled DSv3 step time.
  if "TEWrapper_dot_general/custom_partitioning" not in instruction.metadata.op_name:
    return False
  if "/transpose(" in instruction.metadata.op_name:
    return False
  for computation_id in instruction.called_computation_ids:
    computation = computations_by_id.get(computation_id)
    if computation is not None and any(i.opcode == "all-gather" for i in computation.instructions):
      return True
  return False


def _is_all_gather_start(instruction, computations_by_id) -> bool:
  if instruction.opcode != "async-start":
    return False
  return any(
      computation_id in computations_by_id
      and any(i.opcode == "all-gather" for i in computations_by_id[computation_id].instructions)
      for computation_id in instruction.called_computation_ids
  )


def _transparent_input_closure(start, instructions_by_id: dict[int, object]) -> set[int]:
  """Returns start plus zero-cost input projections which may be hoisted."""
  closure = {start.id}
  worklist = [start]
  while worklist:
    instruction = worklist.pop()
    for operand_id in instruction.operand_ids:
      operand = instructions_by_id.get(operand_id)
      if operand is not None and operand.id not in closure and operand.opcode in _TRANSPARENT_OPCODES:
        closure.add(operand.id)
        worklist.append(operand)
  return closure


def _successors(instructions) -> dict[int, list]:
  successors = defaultdict(list)
  for instruction in instructions:
    for predecessor_id in (*instruction.operand_ids, *instruction.control_predecessor_ids):
      successors[predecessor_id].append(instruction)
  return successors


def _validate_topological_order(sequence: list[int], instructions_by_id: dict[int, object]) -> None:
  positions = {instruction_id: index for index, instruction_id in enumerate(sequence)}
  if len(positions) != len(sequence):
    raise ValueError("HLO schedule contains duplicate instruction ids")
  for instruction_id in sequence:
    instruction = instructions_by_id[instruction_id]
    for predecessor_id in (*instruction.operand_ids, *instruction.control_predecessor_ids):
      predecessor_position = positions.get(predecessor_id)
      if predecessor_position is not None and predecessor_position >= positions[instruction_id]:
        raise ValueError(
            f"Invalid schedule after FSDP overlap pass: {instruction.name} precedes dependency id={predecessor_id}"
        )


def transform_hlo_module(
    module_bytes: bytes, *, max_hoist: int = 128, max_original_window: int = 32
) -> bytes | None:
  """Hoists late FSDP all-gather starts in a serialized HloModuleProto."""
  # TensorFlow packages the XLA protobuf definitions used by JAX/XLA.  Keep the
  # import lazy so normal MaxText runs do not acquire a TensorFlow dependency.
  from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=import-outside-toplevel

  module = hlo_pb2.HloModuleProto()
  module.ParseFromString(module_bytes)
  computations_by_id = {computation.id: computation for computation in module.computations}
  changed = False

  for computation in module.computations:
    sequence_proto = module.schedule.sequences.get(computation.id)
    if sequence_proto is None:
      continue
    instructions_by_id = {instruction.id: instruction for instruction in computation.instructions}
    successors = _successors(computation.instructions)
    sequence = list(sequence_proto.instruction_ids)
    if not sequence:
      continue
    computation_changed = False

    all_gather_starts = [
        instruction
        for instruction in computation.instructions
        if _is_all_gather_start(instruction, computations_by_id)
    ]
    starts = [
        instruction
        for instruction in all_gather_starts
        if _is_fsdp_all_gather_start(instruction, computations_by_id)
    ]
    # Process in launch order and never cross any preceding all-gather start.
    positions = {instruction_id: index for index, instruction_id in enumerate(sequence)}
    all_gather_starts.sort(key=lambda instruction: positions.get(instruction.id, -1))
    starts.sort(key=lambda instruction: positions.get(instruction.id, -1))
    preceding_start = {
        start.id: all_gather_starts[index - 1].id if index else None
        for index, start in enumerate(all_gather_starts)
    }

    for start in starts:
      done_candidates = [user for user in successors.get(start.id, ()) if user.opcode == "async-done"]
      if len(done_candidates) != 1:
        continue
      original_window = positions[done_candidates[0].id] - positions[start.id]
      if original_window > max_original_window:
        continue
      positions = {instruction_id: index for index, instruction_id in enumerate(sequence)}
      movable_ids = _transparent_input_closure(start, instructions_by_id)
      external_predecessors = {
          predecessor_id
          for movable_id in movable_ids
          for predecessor_id in (
              *instructions_by_id[movable_id].operand_ids,
              *instructions_by_id[movable_id].control_predecessor_ids,
          )
          if predecessor_id not in movable_ids and predecessor_id in positions
      }
      earliest_position = 0
      if external_predecessors:
        earliest_position = max(positions[predecessor_id] for predecessor_id in external_predecessors) + 1
      previous_start_id = preceding_start[start.id]
      if previous_start_id is not None:
        earliest_position = max(earliest_position, positions[previous_start_id] + 1)
      start_position = positions[start.id]
      target_position = max(earliest_position, start_position - max_hoist)
      if target_position >= min(positions[instruction_id] for instruction_id in movable_ids):
        continue

      movable_sequence = [instruction_id for instruction_id in sequence if instruction_id in movable_ids]
      remaining_sequence = [instruction_id for instruction_id in sequence if instruction_id not in movable_ids]
      # Translate the old target index after removing movable instructions.
      insertion_position = sum(
          instruction_id not in movable_ids for instruction_id in sequence[:target_position]
      )
      sequence = (
          remaining_sequence[:insertion_position]
          + movable_sequence
          + remaining_sequence[insertion_position:]
      )
      start.frontend_attributes.map["maxtext_fsdp_overlap"] = "hoist_start_with_input_projections"
      changed = True
      computation_changed = True

    if computation_changed:
      _validate_topological_order(sequence, instructions_by_id)
      del sequence_proto.instruction_ids[:]
      sequence_proto.instruction_ids.extend(sequence)

  return module.SerializeToString() if changed else None


def register_fsdp_all_gather_overlap_pass(max_hoist: int = 128, max_original_window: int = 32) -> None:
  """Registers the transformation once for the CUDA backend."""
  global _registered
  if _registered:
    return
  from jax.extend import xla  # pylint: disable=import-outside-toplevel

  xla.register_hlo_module_transformation(
      partial(transform_hlo_module, max_hoist=max_hoist, max_original_window=max_original_window),
      name=_PASS_NAME,
      stage=xla.PipelineStage.POST_SCHEDULER,
      platforms="cuda",
  )
  _registered = True
  max_logging.log(f"Registered {_PASS_NAME} at POST_SCHEDULER")
