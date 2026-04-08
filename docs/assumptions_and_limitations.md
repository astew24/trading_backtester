# Assumptions and Limitations

This project is still a daily-bar backtester. It is more realistic than the original version, but it is not a substitute for an execution simulator built from tick or order-book data.

## Assumptions

- signals are generated on one bar and executed on the next bar's open
- spread and slippage are modeled as simple basis-point adjustments around the execution price
- stops are evaluated using daily OHLC ranges with conservative fill assumptions when multiple exit levels are touched on the same bar
- multi-symbol runs are aggregated as an equal-weighted portfolio of per-symbol strategy returns

## Current limitations

- no borrow availability or hard-to-borrow fee model for shorts
- no portfolio-level cash sharing across symbols beyond equal-weight aggregation
- no intraday event queue or partial-fill model
- Yahoo Finance data quality can vary across symbols and corporate actions

## Next logical steps

- move from per-symbol aggregation to a shared portfolio ledger
- add benchmark-aware reporting and walk-forward validation
- support richer data sources and intraday timestamps
