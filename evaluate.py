from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def ulcer_index(equity: pd.Series) -> float:
    """Path-based drawdown severity: RMS of percentage drawdown over the path.

    Non-gating SECONDARY metric (metric pin, ratified 2026-07-01): max-DD sees
    only the single worst valley; the Ulcer index prices how long and how deep
    the curve spends underwater. Reported alongside, never gating.
    """
    if equity is None or len(equity) == 0:
        return float("nan")
    dd_pct = drawdown(equity.astype(float)) * 100.0
    return float(np.sqrt(np.mean(np.square(dd_pct))))


def summarize_equity(equity_df: pd.DataFrame, initial_equity: float = 10_000.0,
                     periods_per_year: int = 252 * 24 * 12) -> dict:
    """
    periods_per_year for M5 bars: 252 trading days * 24h * 12 bars/h ≈ 72 576.
    XAUUSD trades ~23h/day on 5-day weeks; the default is a conservative upper bound.
    """
    if equity_df.empty:
        return {}
    eq = equity_df["equity"].astype(float)
    n = len(eq)
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = eq.iloc[-1] / initial_equity - 1.0
    max_dd = drawdown(eq).min()

    # Annualised return (CAGR-style scaling by bar count).
    ann_return = (1.0 + total_return) ** (periods_per_year / max(n, 1)) - 1.0

    # Sharpe (excess return over 0-rate risk-free assumption, per-bar scale).
    sharpe = np.nan
    ret_std = returns.std()
    if ret_std and ret_std > 0:
        sharpe = returns.mean() / ret_std * np.sqrt(periods_per_year)

    # Sortino: penalises only downside volatility.
    sortino = np.nan
    down = returns[returns < 0]
    if len(down) > 1:
        down_std = down.std()
        if down_std > 0:
            sortino = returns.mean() / down_std * np.sqrt(periods_per_year)

    # Calmar: annualised return divided by absolute max drawdown.
    calmar = np.nan
    if np.isfinite(ann_return) and abs(max_dd) > 1e-10:
        calmar = ann_return / abs(max_dd)

    # Mark-to-market drawdown (doc 03 §3.9a): includes open-position unrealized
    # PnL, so intra-trade pain is measured — realized-only DD understates it.
    max_dd_mtm = np.nan
    ulcer_mtm = np.nan
    if "equity_mtm" in equity_df.columns:
        max_dd_mtm = drawdown(equity_df["equity_mtm"].astype(float)).min()
        ulcer_mtm = ulcer_index(equity_df["equity_mtm"])

    return {
        "initial_equity": initial_equity,
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "annualized_return_pct": float(ann_return * 100) if np.isfinite(ann_return) else np.nan,
        "max_drawdown_pct": float(max_dd * 100),
        "max_drawdown_mtm_pct": float(max_dd_mtm * 100) if np.isfinite(max_dd_mtm) else np.nan,
        "ulcer_index_mtm": float(ulcer_mtm) if np.isfinite(ulcer_mtm) else np.nan,
        "sharpe_like": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino_ratio": float(sortino) if np.isfinite(sortino) else np.nan,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else np.nan,
        "n_equity_points": int(n),
    }


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "avg_r": np.nan,
            "median_r": np.nan,
            "avg_bars_in_trade": np.nan,
        }
    pnl = trades["pnl"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return {
        "n_trades": int(len(trades)),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": float(pf),
        "avg_r": float(trades["r_mult"].mean()),
        "median_r": float(trades["r_mult"].median()),
        "avg_bars_in_trade": float(trades["bars_in_trade"].mean()),
    }


def trade_based_sharpe(trades: pd.DataFrame, n_bars: int,
                       periods_per_year: int) -> float:
    """Sharpe over per-trade R-multiples, annualised by the realized trade rate.

    Doc 03 §3.9b: the per-bar Sharpe on a mostly-flat realized-equity step
    series (annualised by the full bar count) is noisy and easy to misread.
    Per-trade R-multiples are the natural unit of this system's risk.
    """
    if trades is None or trades.empty or n_bars <= 0:
        return float("nan")
    r = trades["r_mult"].astype(float)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    years = n_bars / float(periods_per_year)
    if not np.isfinite(sd) or sd <= 0 or years <= 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(len(r) / years))


def full_report(equity_df: pd.DataFrame, trades: pd.DataFrame,
                initial_equity: float = 10_000.0,
                periods_per_year: int = 252 * 24 * 12) -> pd.DataFrame:
    d = {}
    d.update(summarize_equity(equity_df, initial_equity=initial_equity,
                              periods_per_year=periods_per_year))
    d.update(summarize_trades(trades))
    d["sharpe_trade"] = trade_based_sharpe(
        trades, n_bars=len(equity_df), periods_per_year=periods_per_year)
    return pd.DataFrame.from_dict(d, orient="index", columns=["value"])
