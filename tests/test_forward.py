import pytest
import torch

from flash_attn_lab import forward
from tests.reference import attention_reference

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available()
                                 or torch.cuda.get_device_capability()[0] != 8,
                                 reason="An Ampere CUDA GPU is required")]


@torch.inference_mode()
def test_forward(dense_case):
    scenario, version, cfg, _ = dense_case
    torch.manual_seed(17)
    q = torch.randn((2, 2, 256, cfg.head_dim), device="cuda", dtype=cfg.dtype.to_torch_dtype())
    k, v = torch.randn_like(q), torch.randn_like(q)
    output = torch.empty_like(q)
    actual = forward(cfg, q, k, v, scenario=scenario, version=version, o=output)
    assert actual.data_ptr() == output.data_ptr()
    expected = attention_reference(q, k, v, causal=scenario == "causal")
    tolerance = 2e-3 if q.dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)

    if scenario == "causal":
        # Split inside a tile to exercise both the tile bound and element masks.
        prefix = cfg.Br + cfg.Bc // 2 + 1
        k[:, :, prefix:] = torch.randn_like(k[:, :, prefix:]) * 5
        v[:, :, prefix:] = torch.randn_like(v[:, :, prefix:]) * 5
        changed = forward(cfg, q, k, v, scenario=scenario, version=version)
        torch.testing.assert_close(changed[:, :, :prefix], actual[:, :, :prefix], atol=0, rtol=0)
