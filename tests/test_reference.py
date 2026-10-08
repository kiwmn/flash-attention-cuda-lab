import torch

from helper.paged_inputs import make_paged_cache
from tests.reference import attention_reference


def test_bottom_right_causal_mask():
    q = torch.zeros(1, 1, 2, 1)
    k = torch.zeros(1, 1, 4, 1)
    v = torch.tensor([1., 2., 3., 10.]).reshape(1, 1, 4, 1)
    actual = attention_reference(q, k, v, causal=True)
    torch.testing.assert_close(actual.flatten(), torch.tensor([2., 4.]))


def test_shuffled_cache_recovers_logical_tokens():
    k = torch.arange(43 * 2 * 8, dtype=torch.float32).reshape(43, 2, 8)
    starts = torch.tensor([0, 17, 43], dtype=torch.int32)
    kc, vc, table = make_paged_cache(k, -k, starts, page_size=16, seed=3)
    for request, (begin, end) in enumerate(zip(starts.tolist(), starts.tolist()[1:])):
        rows = torch.arange(end - begin)
        physical = table[request, rows // 16]
        torch.testing.assert_close(kc[physical, rows % 16], k[begin:end])
        torch.testing.assert_close(vc[physical, rows % 16], -k[begin:end])
    assert torch.isnan(kc[table[0, 1], 1:]).all()
