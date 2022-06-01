import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False, help="plot spectra")
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="run benchmarks against obspy",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "plot: mark test that generates a plot")
    config.addinivalue_line("markers", "benchmark: mark test that run a benchmark")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        return
    if config.getoption("--benchmark"):
        return
    skip_plot = pytest.mark.skip(reason="need --plot option to run")
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    for item in items:
        if "plot" in item.keywords:
            item.add_marker(skip_plot)
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
