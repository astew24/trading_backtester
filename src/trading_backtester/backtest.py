from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_price_data, validate_price_data


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    position_size: float = 0.95
    max_leverage: float = 1.0
    commission_bps: float = 5.0
    fixed_commission: float = 1.0
    spread_bps: float = 2.0
    slippage_bps: float = 1.0
    volatility_target: float | None = None
    volatility_lookback: int = 20
    stop_loss: float | None = 0.08
    take_profit: float | None = 0.15
    trailing_stop: float | None = 0.1
    close_positions_on_finish: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BacktestConfig:
        return cls(**config)


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    side: str
    quantity: float
    entry_price: float
    entry_costs: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_costs: float = 0.0
    exit_reason: str | None = None
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["entry_date"] = self.entry_date.isoformat()
        record["exit_date"] = (
            self.exit_date.isoformat() if self.exit_date is not None else None
        )
        return record


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.cash = self.config.initial_capital
        self.shares = 0.0
        self.active_trade: Trade | None = None
        self.high_watermark: float | None = None
        self.low_watermark: float | None = None

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        *,
        symbol: str = "asset",
    ) -> tuple[pd.DataFrame, list[Trade]]:
        price_data = validate_price_data(data)
        signal_frame = self._prepare_signals(signals, price_data.index)
        volatility = self._realized_volatility(price_data["Close"], signal_frame)

        self.cash = self.config.initial_capital
        self.shares = 0.0
        self.active_trade = None
        self.high_watermark = None
        self.low_watermark = None
        last_equity = self.config.initial_capital
        trades: list[Trade] = []
        rows: list[dict[str, Any]] = []

        for index, (date, bar) in enumerate(price_data.iterrows()):
            open_price = float(bar["Open"])
            close_price = float(bar["Close"])
            turnover = 0.0

            # Signal is taken from the previous bar (index - 1) to reflect end-of-day
            # generation; fills execute at today's open. This is the key lookahead guard.
            target_signal = (
                float(signal_frame["target_position"].iloc[index - 1])
                if index > 0
                else 0.0
            )

            if index > 0:
                if self.shares != 0.0 and self._should_flip_or_flatten(target_signal):
                    turnover += self._close_position(
                        date=date,
                        raw_price=open_price,
                        symbol=symbol,
                        trades=trades,
                        reason="signal",
                    )

                if self.shares == 0.0 and target_signal != 0.0:
                    turnover += self._open_position(
                        date=date,
                        raw_price=open_price,
                        symbol=symbol,
                        direction=np.sign(target_signal),
                        equity=last_equity,
                        realized_vol=volatility.iloc[index - 1],
                    )

                if self.shares != 0.0:
                    turnover += self._apply_intraday_exits(
                        date=date,
                        bar=bar,
                        symbol=symbol,
                        trades=trades,
                    )

            market_value = self.shares * close_price
            equity = self.cash + market_value
            gross_exposure = abs(market_value)
            net_exposure = market_value
            returns = (equity / last_equity) - 1.0 if rows else 0.0
            rows.append(
                {
                    "Date": date,
                    "Cash": self.cash,
                    "Shares": self.shares,
                    "Market_Value": market_value,
                    "Gross_Exposure": gross_exposure,
                    "Net_Exposure": net_exposure,
                    "Portfolio_Value": equity,
                    "Returns": returns,
                    "Signal": target_signal,
                    "Position": net_exposure / equity if equity else 0.0,
                    "Turnover": turnover / max(last_equity, 1.0),
                }
            )
            last_equity = equity

        results = pd.DataFrame(rows).set_index("Date")

        if self.config.close_positions_on_finish and self.shares != 0.0:
            final_date = price_data.index[-1]
            final_close = float(price_data["Close"].iloc[-1])
            final_turnover = self._close_position(
                date=final_date,
                raw_price=final_close,
                symbol=symbol,
                trades=trades,
                reason="end_of_test",
            )
            results.iloc[-1, results.columns.get_loc("Cash")] = self.cash
            results.iloc[-1, results.columns.get_loc("Shares")] = self.shares
            results.iloc[-1, results.columns.get_loc("Market_Value")] = 0.0
            results.iloc[-1, results.columns.get_loc("Gross_Exposure")] = 0.0
            results.iloc[-1, results.columns.get_loc("Net_Exposure")] = 0.0
            results.iloc[-1, results.columns.get_loc("Portfolio_Value")] = self.cash
            results.iloc[-1, results.columns.get_loc("Position")] = 0.0
            prior_value = (
                float(results["Portfolio_Value"].iloc[-2])
                if len(results) > 1
                else self.config.initial_capital
            )
            turnover_index = results.columns.get_loc("Turnover")
            results.iloc[-1, turnover_index] += final_turnover / max(prior_value, 1.0)
            results["Returns"] = results["Portfolio_Value"].pct_change().fillna(0.0)

        return results, trades

    def _prepare_signals(self, signals: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        signal_frame = signals.copy()
        if "target_position" not in signal_frame.columns:
            for candidate in ("Position", "position", "Signal", "signal"):
                if candidate in signal_frame.columns:
                    signal_frame["target_position"] = signal_frame[candidate]
                    break
        if "target_position" not in signal_frame.columns:
            raise ValueError("Signals must contain a target_position column")
        signal_frame = signal_frame.reindex(index).ffill().fillna(0.0)
        signal_frame["target_position"] = signal_frame["target_position"].clip(
            lower=-self.config.max_leverage,
            upper=self.config.max_leverage,
        )
        return signal_frame

    def _realized_volatility(
        self, close: pd.Series, signals: pd.DataFrame
    ) -> pd.Series:
        if "realized_volatility" in signals.columns:
            return pd.to_numeric(
                signals["realized_volatility"], errors="coerce"
            ).ffill()
        daily_returns = close.pct_change()
        return daily_returns.rolling(self.config.volatility_lookback).std(
            ddof=0
        ) * np.sqrt(252)

    def _position_fraction(
        self, signal_strength: float, realized_vol: float | None
    ) -> float:
        fraction = self.config.position_size * abs(signal_strength)
        if self.config.volatility_target and realized_vol and realized_vol > 0:
            fraction = min(fraction, self.config.volatility_target / realized_vol)
        return float(min(max(fraction, 0.0), self.config.max_leverage))

    def _fill_price(self, raw_price: float, direction: float) -> float:
        half_spread = self.config.spread_bps / 20000.0
        slippage = self.config.slippage_bps / 10000.0
        return raw_price * (1.0 + (direction * half_spread) + (direction * slippage))

    def _commission(self, notional: float) -> float:
        if notional <= 0.0:
            return 0.0
        variable = notional * (self.config.commission_bps / 10000.0)
        return variable + self.config.fixed_commission

    def _should_flip_or_flatten(self, target_signal: float) -> bool:
        if self.shares == 0.0:
            return False
        if target_signal == 0.0:
            return True
        return np.sign(target_signal) != np.sign(self.shares)

    def _open_position(
        self,
        *,
        date: pd.Timestamp,
        raw_price: float,
        symbol: str,
        direction: float,
        equity: float,
        realized_vol: float | None,
    ) -> float:
        fraction = self._position_fraction(direction, realized_vol)
        if fraction == 0.0:
            return 0.0
        quantity = (equity * fraction) / raw_price
        quantity *= direction
        fill_price = self._fill_price(raw_price, np.sign(quantity))
        notional = abs(quantity * fill_price)
        costs = self._commission(notional)
        self.cash -= quantity * fill_price
        self.cash -= costs
        self.shares += quantity
        self.active_trade = Trade(
            symbol=symbol,
            entry_date=date,
            side="long" if quantity > 0 else "short",
            quantity=abs(quantity),
            entry_price=fill_price,
            entry_costs=costs,
        )
        self.high_watermark = fill_price
        self.low_watermark = fill_price
        return notional

    def _close_position(
        self,
        *,
        date: pd.Timestamp,
        raw_price: float,
        symbol: str,
        trades: list[Trade],
        reason: str,
    ) -> float:
        if self.shares == 0.0 or self.active_trade is None:
            return 0.0
        quantity = -self.shares
        fill_price = self._fill_price(raw_price, np.sign(quantity))
        notional = abs(quantity * fill_price)
        costs = self._commission(notional)
        self.cash -= quantity * fill_price
        self.cash -= costs
        self.shares = 0.0

        trade = self.active_trade
        trade.exit_date = date
        trade.exit_price = fill_price
        trade.exit_costs = costs
        trade.exit_reason = reason
        direction = 1.0 if trade.side == "long" else -1.0
        trade.gross_pnl = (fill_price - trade.entry_price) * trade.quantity * direction
        trade.net_pnl = trade.gross_pnl - trade.entry_costs - trade.exit_costs
        entry_notional = trade.entry_price * trade.quantity
        trade.return_pct = trade.net_pnl / entry_notional if entry_notional else 0.0
        trades.append(trade)
        self.active_trade = None
        self.high_watermark = None
        self.low_watermark = None
        return notional

    def _apply_intraday_exits(
        self,
        *,
        date: pd.Timestamp,
        bar: pd.Series,
        symbol: str,
        trades: list[Trade],
    ) -> float:
        if self.active_trade is None or self.shares == 0.0:
            return 0.0

        open_price = float(bar["Open"])
        high_price = float(bar["High"])
        low_price = float(bar["Low"])
        entry_price = self.active_trade.entry_price
        direction = 1.0 if self.shares > 0 else -1.0

        triggered: list[tuple[str, float]] = []

        if direction > 0:
            self.high_watermark = max(self.high_watermark or entry_price, high_price)
            if self.config.stop_loss is not None:
                level = entry_price * (1.0 - self.config.stop_loss)
                if low_price <= level:
                    triggered.append(("stop_loss", min(open_price, level)))
            if self.config.take_profit is not None:
                level = entry_price * (1.0 + self.config.take_profit)
                if high_price >= level:
                    triggered.append(("take_profit", max(open_price, level)))
            if (
                self.config.trailing_stop is not None
                and self.high_watermark is not None
            ):
                level = self.high_watermark * (1.0 - self.config.trailing_stop)
                if low_price <= level:
                    triggered.append(("trailing_stop", min(open_price, level)))
        else:
            self.low_watermark = min(self.low_watermark or entry_price, low_price)
            if self.config.stop_loss is not None:
                level = entry_price * (1.0 + self.config.stop_loss)
                if high_price >= level:
                    triggered.append(("stop_loss", max(open_price, level)))
            if self.config.take_profit is not None:
                level = entry_price * (1.0 - self.config.take_profit)
                if low_price <= level:
                    triggered.append(("take_profit", min(open_price, level)))
            if self.config.trailing_stop is not None and self.low_watermark is not None:
                level = self.low_watermark * (1.0 + self.config.trailing_stop)
                if high_price >= level:
                    triggered.append(("trailing_stop", max(open_price, level)))

        if not triggered:
            return 0.0

        worst_reason, worst_price = self._select_conservative_exit(triggered, direction)
        return self._close_position(
            date=date,
            raw_price=worst_price,
            symbol=symbol,
            trades=trades,
            reason=worst_reason,
        )

    def _select_conservative_exit(
        self,
        exits: list[tuple[str, float]],
        direction: float,
    ) -> tuple[str, float]:
        if direction > 0:
            return min(exits, key=lambda item: item[1])
        return max(exits, key=lambda item: item[1])


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([trade.to_record() for trade in trades])


def run_backtest(
    data_path: str | Path,
    signals_path: str | Path,
    *,
    initial_capital: float = 100000.0,
) -> pd.DataFrame:
    price_data = load_price_data(data_path)
    signals = pd.read_csv(signals_path, index_col=0, parse_dates=True)
    engine = BacktestEngine(BacktestConfig(initial_capital=initial_capital))
    results, _ = engine.run(price_data, signals)
    return results
