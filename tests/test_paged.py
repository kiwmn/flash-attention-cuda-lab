import pytest
import torch

from flash_attn_lab import paged_attention
from helper.paged_inputs import make_paged_cache
from tests.reference import packed_attention_reference

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available()
                                 or torch.cuda.get_device_capability()[0] != 8,
                                 reason="An Ampere CUDA GPU is required")]


@pytest.mark.parametrize("kv_heads", [1, 2, 4], ids=["mqa", "gqa", "mha"])
@torch.inference_mode()
def test_paged(paged_case, kv_heads):
    _, version, cfg, page_size = paged_case
    torch.manual_seed(43)
    q_lens = [1, 17, 65, 129]
    kv_lens = [19, 17, 143, 291]
    query_starts = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()],
                                device="cuda", dtype=torch.int32)
    kv_starts = torch.tensor([0, *torch.tensor(kv_lens).cumsum(0).tolist()],
                             device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    q = torch.randn((sum(q_lens), 4, cfg.head_dim), device="cuda", dtype=cfg.dtype.to_torch_dtype())
    k = torch.randn((sum(kv_lens), kv_heads, cfg.head_dim), device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    k_cache, v_cache, table = make_paged_cache(k, v, kv_starts, page_size, seed=7)
    output = torch.empty_like(q)
    actual = paged_attention(
        cfg, q, k_cache, v_cache, query_starts, seq_lens, table,
        max_query_len=max(q_lens), max_kv_len=max(kv_lens), version=version,
        o=output, validate_metadata=True,
    )
    expected = packed_attention_reference(q, k, v, query_starts, kv_starts)
    assert actual.data_ptr() == output.data_ptr()
    tolerance = 2e-3 if q.dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
