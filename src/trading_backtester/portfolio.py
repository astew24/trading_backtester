from __future__ import annotations

import pandas as pd


def buy_and_hold_curve(close: pd.Series, initial_capital: float) -> pd.Series:
    returns = pd.to_numeric(close, errors="coerce").pct_change().fillna(0.0)
    return initial_capital * (1.0 + returns).cumprod()


def combine_results(
    results_by_symbol: dict[str, pd.DataFrame], initial_capital: float
) -> pd.DataFrame:
    """Combine per-symbol result frames into an equal-weighted portfolio.

    Each symbol contributes an equal share of ``initial_capital``. Returns are
    averaged across symbols on each date; portfolio value compounds from there.
    """
    if not results_by_symbol:
        raise ValueError("results_by_symbol cannot be empty")

    combined_returns = (
        pd.concat(
            [
                frame["Returns"].rename(symbol)
                for symbol, frame in results_by_symbol.items()
            ],
            axis=1,
        )
        .fillna(0.0)
        .mean(axis=1)
    )
    combined = pd.DataFrame(index=combined_returns.index)
    combined["Returns"] = combined_returns
    combined["Portfolio_Value"] = initial_capital * (1.0 + combined_returns).cumprod()
    combined["Cash"] = combined["Portfolio_Value"]
    combined["Shares"] = 0.0
    combined["Market_Value"] = 0.0
    combined["Gross_Exposure"] = (
        pd.concat(
            [
                frame["Gross_Exposure"].rename(symbol)
                for symbol, frame in results_by_symbol.items()
            ],
            axis=1,
        )
        .fillna(0.0)
        .mean(axis=1)
    )
    combined["Net_Exposure"] = (
        pd.concat(
            [
                frame["Net_Exposure"].rename(symbol)
                for symbol, frame in results_by_symbol.items()
            ],
            axis=1,
        )
        .fillna(0.0)
        .mean(axis=1)
    )
    combined["Position"] = combined["Net_Exposure"] / combined[
        "Portfolio_Value"
    ].replace(0.0, pd.NA)
    combined["Turnover"] = (
        pd.concat(
            [
                frame["Turnover"].rename(symbol)
                for symbol, frame in results_by_symbol.items()
            ],
            axis=1,
        )
        .fillna(0.0)
        .mean(axis=1)
    )
    combined["Signal"] = 0.0
    return combined.fillna(0.0)


def rebuild_results_from_ratios(
    *,
    returns: pd.Series,
    gross_exposure_ratio: pd.Series,
    net_exposure_ratio: pd.Series,
    turnover: pd.Series,
    initial_capital: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=returns.index)
    frame["Returns"] = returns.fillna(0.0)
    frame["Portfolio_Value"] = initial_capital * (1.0 + frame["Returns"]).cumprod()
    gross_ratio = gross_exposure_ratio.reindex(frame.index).fillna(0.0)
    net_ratio = net_exposure_ratio.reindex(frame.index).fillna(0.0)
    frame["Gross_Exposure"] = gross_ratio * frame["Portfolio_Value"]
    frame["Net_Exposure"] = net_ratio * frame["Portfolio_Value"]
    frame["Market_Value"] = frame["Net_Exposure"]
    frame["Cash"] = frame["Portfolio_Value"] - frame["Market_Value"]
    frame["Shares"] = 0.0
    frame["Position"] = net_ratio
    frame["Turnover"] = turnover.reindex(frame.index).fillna(0.0)
    frame["Signal"] = 0.0
    return frame.fillna(0.0)
