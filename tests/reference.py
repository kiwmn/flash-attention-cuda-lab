"""Small FP32 reference implementations used only for numerical validation."""

import torch


def attention_reference(q, k, v, *, causal=False):
    """Return FP32 attention for [B,H,N,D], including bottom-right causal masks."""
    q, k, v = q.float(), k.float(), v.float()
    if q.shape[-3] != k.shape[-3]:
        groups = q.shape[-3] // k.shape[-3]
        k = k.repeat_interleave(groups, dim=-3)
        v = v.repeat_interleave(groups, dim=-3)
    scores = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
    if causal:
        query_len, kv_len = scores.shape[-2:]
        query_pos = torch.arange(query_len, device=q.device) + kv_len - query_len
        key_pos = torch.arange(kv_len, device=q.device)
        scores.masked_fill_(key_pos[None, :] > query_pos[:, None], -torch.inf)
    return scores.softmax(dim=-1) @ v


def packed_attention_reference(q, k, v, query_starts, kv_starts):
    """Use logical packed KV; page packing is intentionally outside the oracle."""
    query_starts = query_starts.cpu().tolist()
    kv_starts = kv_starts.cpu().tolist()
    outputs = []
    for qb, qe, kb, ke in zip(query_starts, query_starts[1:], kv_starts, kv_starts[1:]):
        result = attention_reference(
            q[qb:qe].transpose(0, 1).unsqueeze(0),
            k[kb:ke].transpose(0, 1).unsqueeze(0),
            v[kb:ke].transpose(0, 1).unsqueeze(0), causal=True,
        )
        outputs.append(result.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)
