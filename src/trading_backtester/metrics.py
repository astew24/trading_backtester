from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Trade


@dataclass
class PerformanceMetrics:
    returns: pd.Series
    risk_free_rate: float = 0.02
    annual_factor: int = 252

    def __post_init__(self) -> None:
        cleaned = pd.to_numeric(self.returns, errors="coerce").fillna(0.0)
        if cleaned.empty:
            raise ValueError("returns series is empty")
        self.returns = cleaned

    def total_return(self) -> float:
        return float((1.0 + self.returns).prod() - 1.0)

    def annualized_return(self) -> float:
        periods = len(self.returns)
        if periods == 0:
            return float("nan")
        compounded = 1.0 + self.total_return()
        if compounded <= 0.0:
            return float("nan")
        years = periods / self.annual_factor
        return float(compounded ** (1.0 / years) - 1.0)

    def annualized_volatility(self) -> float:
        return float(self.returns.std(ddof=0) * np.sqrt(self.annual_factor))

    def sharpe_ratio(self) -> float:
        excess_returns = self.returns - (self.risk_free_rate / self.annual_factor)
        volatility = excess_returns.std(ddof=0)
        if volatility == 0.0:
            return float("nan")
        return float(np.sqrt(self.annual_factor) * excess_returns.mean() / volatility)

    def sortino_ratio(self) -> float:
        excess_returns = self.returns - (self.risk_free_rate / self.annual_factor)
        downside = excess_returns[excess_returns < 0.0]
        downside_deviation = (
            np.sqrt((downside**2).mean()) if not downside.empty else 0.0
        )
        if downside_deviation == 0.0:
            return float("inf")
        return float(
            np.sqrt(self.annual_factor) * excess_returns.mean() / downside_deviation
        )

    def max_drawdown(self) -> float:
        equity_curve = (1.0 + self.returns).cumprod()
        rolling_peak = equity_curve.cummax()
        drawdown = (equity_curve / rolling_peak) - 1.0
        return float(drawdown.min())

    def calmar_ratio(self) -> float:
        drawdown = self.max_drawdown()
        if drawdown == 0.0:
            return float("inf")
        return float(self.annualized_return() / abs(drawdown))


def _trade_metrics(trades: Iterable[Trade]) -> dict[str, float]:
    trade_list = list(trades)
    if not trade_list:
        return {
            "Win Rate": float("nan"),
            "Profit Factor": float("nan"),
            "Average Trade Return": float("nan"),
            "Total Trades": 0.0,
        }

    net_pnls = np.array([trade.net_pnl for trade in trade_list], dtype=float)
    returns = np.array([trade.return_pct for trade in trade_list], dtype=float)
    gross_profit = net_pnls[net_pnls > 0.0].sum()
    gross_loss = abs(net_pnls[net_pnls < 0.0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else float("inf")

    return {
        "Win Rate": float((net_pnls > 0.0).mean()),
        "Profit Factor": float(profit_factor),
        "Average Trade Return": float(np.nanmean(returns)),
        "Total Trades": float(len(trade_list)),
    }


def calculate_metrics(
    data: pd.Series | pd.DataFrame,
    trades: Iterable[Trade] | None = None,
    *,
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    if isinstance(data, pd.DataFrame):
        if "Returns" not in data.columns:
            raise ValueError("Results frame must include a Returns column")
        returns = data["Returns"]
        avg_turnover = (
            float(data["Turnover"].mean())
            if "Turnover" in data.columns
            else float("nan")
        )
        avg_gross_exposure = (
            float(
                (
                    data["Gross_Exposure"]
                    / data["Portfolio_Value"].replace(0.0, np.nan)
                ).mean()
            )
            if {"Gross_Exposure", "Portfolio_Value"}.issubset(data.columns)
            else float("nan")
        )
    else:
        returns = data
        avg_turnover = float("nan")
        avg_gross_exposure = float("nan")

    stats = PerformanceMetrics(returns=returns, risk_free_rate=risk_free_rate)
    metrics = {
        "Total Return": stats.total_return(),
        "Annualized Return": stats.annualized_return(),
        "Annualized Volatility": stats.annualized_volatility(),
        "Sharpe Ratio": stats.sharpe_ratio(),
        "Sortino Ratio": stats.sortino_ratio(),
        "Max Drawdown": stats.max_drawdown(),
        "Calmar Ratio": stats.calmar_ratio(),
        "Average Daily Turnover": avg_turnover,
        "Average Gross Exposure": avg_gross_exposure,
    }
    metrics.update(_trade_metrics(trades or []))
    return metrics
