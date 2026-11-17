import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def gather_worker(rank, rendezvous, result_path, dtype):
    source = Path(__file__).resolve().parents[1] / "src/myvllm/layers/embedding_head.py"
    spec = importlib.util.spec_from_file_location("embedding_head_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        torch.set_default_dtype(torch.float32)
        head = module.ParallelLMHead(5, 2).to(dtype=dtype)
        weights = torch.arange(10, dtype=dtype).reshape(5, 2)
        head.weight_loader(head.weight, weights)
        inputs = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        with torch.no_grad():
            output = head(inputs)
        if rank == 0:
            torch.testing.assert_close(output, inputs @ weights.T)
            assert output.dtype == dtype
            assert output.shape == (2, 5)
            torch.save(output, result_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gather_preserves_logits_dtype_and_trims_padding(tmp_path, dtype):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    result_path = str(tmp_path / "result.pt")
    mp.spawn(gather_worker, args=(rendezvous, result_path, dtype), nprocs=2, join=True)
    assert torch.load(result_path, weights_only=True).dtype == dtype
