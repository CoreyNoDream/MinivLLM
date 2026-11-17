import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_append_keeps_existing_full_blocks_unchanged(block_size):
    manager = BlockManager(4, block_size)
    seq = Sequence(list(range(2 * block_size)), block_size=block_size)
    manager.allocate(seq)
    original = [(manager.blocks[i].hash, list(manager.blocks[i].token_ids)) for i in seq.block_table]

    seq.append_token(99)
    assert manager.can_append(seq)
    manager.append(seq)

    assert len(seq.block_table) == 3
    assert [(manager.blocks[i].hash, manager.blocks[i].token_ids) for i in seq.block_table[:2]] == original
    if block_size == 1:
        last = manager.blocks[seq.block_table[-1]]
        assert last.token_ids == [99]
        assert last.hash != -1


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_new_block_requires_free_capacity(block_size):
    manager = BlockManager(1, block_size)
    seq = Sequence(list(range(block_size)), block_size=block_size)
    manager.allocate(seq)
    seq.append_token(99)

    assert not manager.can_append(seq)


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_repeated_decode_allocations_are_released(block_size):
    manager = BlockManager(5, block_size)
    seq = Sequence(list(range(block_size)), block_size=block_size)
    manager.allocate(seq)
    for token in range(100, 100 + 2 * block_size):
        seq.append_token(token)
        assert manager.can_append(seq)
        manager.append(seq)
        assert len(seq.block_table) == seq.num_blocks
    manager.deallocate(seq)

    assert not manager.used_block_ids
    assert len(manager.free_block_ids) == 5
    assert all(block.ref_count == 0 for block in manager.blocks)
