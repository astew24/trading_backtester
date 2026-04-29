import math

import numpy as np
import pandas as pd
from trading_backtester.metrics import PerformanceMetrics, calculate_metrics


def test_performance_metrics_calculate_drawdown_and_sharpe() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])
    metrics = PerformanceMetrics(returns=returns, risk_free_rate=0.0)

    expected_sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)

    assert math.isclose(metrics.max_drawdown(), -0.20, rel_tol=1e-9)
    assert math.isclose(metrics.sharpe_ratio(), expected_sharpe, rel_tol=1e-9)


def test_calculate_metrics_uses_results_frame() -> None:
    results = pd.DataFrame(
        {
            "Returns": [0.0, 0.01, -0.02, 0.015],
            "Turnover": [0.0, 0.1, 0.0, 0.2],
            "Gross_Exposure": [0.0, 95000.0, 90000.0, 0.0],
            "Portfolio_Value": [100000.0, 101000.0, 98980.0, 100464.7],
        }
    )

    metrics = calculate_metrics(results, risk_free_rate=0.0)

    assert "Average Daily Turnover" in metrics
    assert math.isclose(metrics["Average Daily Turnover"], 0.075, rel_tol=1e-9)


def test_calculate_metrics_includes_benchmark_comparison_stats() -> None:
    results = pd.DataFrame(
        {
            "Returns": [0.0, 0.01, -0.005, 0.015],
            "Turnover": [0.0, 0.1, 0.0, 0.1],
            "Gross_Exposure": [0.0, 80000.0, 85000.0, 82000.0],
            "Portfolio_Value": [100000.0, 101000.0, 100495.0, 102002.425],
        }
    )
    benchmark_returns = pd.Series([0.0, 0.008, -0.002, 0.01])

    metrics = calculate_metrics(
        results,
        risk_free_rate=0.0,
        benchmark_returns=benchmark_returns,
    )

    assert "Benchmark Return" in metrics
    assert "Information Ratio" in metrics
    assert "Beta" in metrics
    assert math.isfinite(metrics["Benchmark Return"])
