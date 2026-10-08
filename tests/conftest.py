import pytest

from helper.kernel_configs import get_configs, list_versions


def pytest_addoption(parser):
    parser.addoption("--scenario", choices=("all", "standard", "a100", "causal", "paged"),
                     default="all", help="Kernel scenario to validate")
    parser.addoption("--kernel-version", default="final",
                     help="final, all, or a retained version such as k04")
    parser.addoption("--all-configs", action="store_true",
                     help="Validate every registered configuration")
    parser.addoption("--head-dim", type=int, choices=(64, 128), default=128)


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires an Ampere CUDA GPU")
    selected = config.getoption("--kernel-version")
    if selected not in ("final", "all"):
        try:
            version = int(selected.lower().removeprefix("k"))
        except ValueError as exc:
            raise pytest.UsageError("--kernel-version must be final, all, or kNN") from exc
        requested = config.getoption("--scenario")
        scenarios = ("standard", "a100", "causal", "paged") if requested == "all" else (requested,)
        if not any(version in list_versions(scenario) for scenario in scenarios):
            raise pytest.UsageError(f"no {selected} version exists for scenario {requested}")


def _versions(scenario, selection):
    if selection == "final":
        return [None]
    if selection == "all":
        return [None, *list_versions(scenario)]
    try:
        version = int(selection.lower().removeprefix("k"))
    except ValueError as exc:
        raise pytest.UsageError("--kernel-version must be final, all, or kNN") from exc
    if version not in list_versions(scenario):
        return []
    return [version]


def representative_configs(configs):
    """Cover resident/streamed operands and buffering without a full sweep."""
    if not configs:
        return []
    selected = [configs[0]]
    for predicate in (
        lambda c: c.prefetch_kv_tiles and not c.mma_double_buffer_loads,
        lambda c: c.mma_double_buffer_loads and c.optimized_softmax,
    ):
        cfg = next((c for c in configs if predicate(c)), None)
        if cfg is not None and cfg not in selected:
            selected.append(cfg)
    return selected


def pytest_generate_tests(metafunc):
    fixture = next((f for f in ("dense_case", "paged_case") if f in metafunc.fixturenames), None)
    if fixture is None:
        return
    requested = metafunc.config.getoption("--scenario")
    scenarios = ["paged"] if fixture == "paged_case" else ["standard", "a100", "causal"]
    cases, ids = [], []
    for scenario in scenarios:
        if requested not in ("all", scenario):
            continue
        for version in _versions(scenario, metafunc.config.getoption("--kernel-version")):
            for dtype in ("fp16", "bf16"):
                pages = (16, 256) if scenario == "paged" else (None,)
                for page in pages:
                    configs = get_configs(scenario, version=version, dtype=dtype,
                                          head_dim=metafunc.config.getoption("--head-dim"),
                                          page_size=page)
                    if not metafunc.config.getoption("--all-configs"):
                        configs = representative_configs(configs)
                    for cfg in configs:
                        version_name = "final" if version is None else f"k{version:02d}"
                        ids.append(f"{scenario}-{version_name}-{cfg}-page{page}")
                        cases.append((scenario, version, cfg, page))
    metafunc.parametrize(fixture, cases, ids=ids)
