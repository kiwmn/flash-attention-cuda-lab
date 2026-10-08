"""Construct packed requests and physically shuffled, read-only KV caches."""

import torch


def make_inverse_q_history_lengths(total_tokens, num_requests, device="cuda"):
    """Keep KV length fixed while increasing Q length across requests."""
    if total_tokens <= 0 or num_requests <= 0 or total_tokens % num_requests:
        raise ValueError("total_tokens must be positive and divisible by num_requests")
    kv_len = total_tokens // num_requests
    if kv_len < num_requests:
        raise ValueError("KV length must be at least num_requests")
    index = torch.arange(1, num_requests + 1, device=device, dtype=torch.int64)
    q_lens = (kv_len * index + num_requests - 1) // num_requests
    seq_lens = torch.full_like(q_lens, kv_len)
    starts = torch.nn.functional.pad(q_lens.cumsum(0), (1, 0))
    return tuple(x.to(torch.int32) for x in (q_lens, seq_lens - q_lens, seq_lens, starts))


def make_paged_cache(k, v, kv_start_loc, page_size, *, seed=0):
    """Pack logical KV into shuffled pages; poison unused entries with NaNs."""
    if page_size <= 0 or k.shape != v.shape or k.ndim != 3:
        raise ValueError("expected matching [tokens, heads, dim] K/V and positive page_size")
    starts = kv_start_loc.cpu().tolist()
    lengths = [b - a for a, b in zip(starts, starts[1:])]
    if not lengths or starts[0] != 0 or starts[-1] != k.shape[0] or min(lengths) <= 0:
        raise ValueError("KV offsets must partition the input into nonempty requests")
    counts = [(n + page_size - 1) // page_size for n in lengths]
    pages = sum(counts)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(pages, generator=generator).tolist()
    shape = (pages, page_size, k.shape[1], k.shape[2])
    k_cache = torch.full(shape, torch.nan, device=k.device, dtype=k.dtype)
    v_cache = torch.full_like(k_cache, torch.nan)
    block_table = torch.full((len(lengths), max(counts)), -1,
                            device=k.device, dtype=torch.int32)
    cursor = 0
    for request, length in enumerate(lengths):
        for logical_page in range(counts[request]):
            physical_page = permutation[cursor]
            cursor += 1
            begin = logical_page * page_size
            valid = min(page_size, length - begin)
            block_table[request, logical_page] = physical_page
            start = starts[request] + begin
            k_cache[physical_page, :valid] = k[start:start + valid]
            v_cache[physical_page, :valid] = v[start:start + valid]
    return k_cache, v_cache, block_table
