"""Enhanced Heatmap Engine — multi-timeframe, multi-domain ANSI heatmaps.

Provides:
    - Enhanced sector heatmap with ANSI colors
    - Multi-timeframe heatmaps (1D, 1W, 1M, 3M)
    - Factor heatmap (momentum, value, quality, volatility)
    - Correlation heatmap between sectors
    - Portfolio exposure heatmap
    - International market heatmap (US, EU, Asia)
    - ASCII art rendering with ANSI color codes
"""

import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..data.yahoo_data import get_adj_close, get_returns
from ..data.universe_engine import (
    SECTOR_ETFS, FACTOR_ETFS, FI_ETFS, COMMODITY_ETFS,
    INTL_ETFS, VOL_ETFS,
)


# ---------------------------------------------------------------------------
# ANSI colors & rendering constants
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[97m"

# Extended ANSI 256-color palette for fine-grained heatmaps
# Red gradient (negative returns)
_RED_GRADIENT = [
    "\033[38;5;52m",   # darkest red
    "\033[38;5;88m",
    "\033[38;5;124m",
    "\033[38;5;160m",
    "\033[38;5;196m",  # bright red
]

# Green gradient (positive returns)
_GREEN_GRADIENT = [
    "\033[38;5;22m",   # dark green
    "\033[38;5;28m",
    "\033[38;5;34m",
    "\033[38;5;40m",
    "\033[38;5;46m",   # bright green
]

# Neutral
_NEUTRAL_COLOR = "\033[38;5;240m"  # gray


# Block characters for heatmap cells
BLOCK_FULL = "██"
BLOCK_DENSE = "▓▓"
BLOCK_MEDIUM = "▒▒"
BLOCK_LIGHT = "░░"
BLOCK_EMPTY = "  "


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def _return_to_color(ret: float, scale: float = 0.03) -> str:
    """Map a return value to an ANSI color string.

    Args:
        ret: return value (e.g. 0.01 = 1%)
        scale: the return magnitude mapped to the brightest color

    Returns:
        ANSI color escape code.
    """
    if abs(ret) < 0.001:
        return _NEUTRAL_COLOR

    intensity = min(abs(ret) / scale, 1.0)
    index = int(intensity * (len(_GREEN_GRADIENT) - 1))
    index = max(0, min(index, len(_GREEN_GRADIENT) - 1))

    if ret > 0:
        return _GREEN_GRADIENT[index]
    else:
        return _RED_GRADIENT[index]


def _return_to_block(ret: float, scale: float = 0.03) -> str:
    """Map a return to a colored block character."""
    color = _return_to_color(ret, scale)
    intensity = min(abs(ret) / scale, 1.0)

    if intensity > 0.75:
        block = BLOCK_FULL
    elif intensity > 0.50:
        block = BLOCK_DENSE
    elif intensity > 0.25:
        block = BLOCK_MEDIUM
    elif intensity > 0.05:
        block = BLOCK_LIGHT
    else:
        block = BLOCK_EMPTY

    return f"{color}{block}{RESET}"


def _corr_to_color(corr: float) -> str:
    """Map a correlation value [-1, 1] to an ANSI color."""
    if corr > 0.7:
        return "\033[38;5;46m"   # bright green
    elif corr > 0.4:
        return "\033[38;5;34m"   # green
    elif corr > 0.1:
        return "\033[38;5;28m"   # dark green
    elif corr > -0.1:
        return _NEUTRAL_COLOR
    elif corr > -0.4:
        return "\033[38;5;160m"  # red
    elif corr > -0.7:
        return "\033[38;5;196m"  # bright red
    else:
        return "\033[38;5;196m"  # brightest red


# ---------------------------------------------------------------------------
# Timeframe definitions
# ---------------------------------------------------------------------------
TIMEFRAME_DAYS = {
    "1D": 2,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
}


# ---------------------------------------------------------------------------
# International market tickers
# ---------------------------------------------------------------------------
INTERNATIONAL_MARKETS = {
    "US": {
        "S&P 500": "SPY",
        "NASDAQ 100": "QQQ",
        "Russell 2000": "IWM",
        "Dow Jones": "DIA",
    },
    "Europe": {
        "Europe (Broad)": "VGK",
        "UK (FTSE)": "EWU",
        "Germany (DAX)": "EWG",
        "France (CAC)": "EWQ",
    },
    "Asia": {
        "Japan (Nikkei)": "EWJ",
        "China (FXI)": "FXI",
        "Hong Kong": "EWH",
        "South Korea": "EWY",
        "India": "INDA",
    },
    "Emerging": {
        "Emerging Mkts": "EEM",
        "Brazil": "EWZ",
        "Mexico": "EWW",
        "Taiwan": "EWT",
    },
}


# ---------------------------------------------------------------------------
# Core heatmap data
# ---------------------------------------------------------------------------
@dataclass
class HeatmapCell:
    """Single cell in a heatmap."""
    label: str
    value: float = 0.0
    color: str = ""
    block: str = ""
    formatted: str = ""


@dataclass
class HeatmapGrid:
    """A complete heatmap grid."""
    title: str
    rows: list = field(default_factory=list)       # list of row labels
    columns: list = field(default_factory=list)     # list of column labels
    cells: dict = field(default_factory=dict)       # (row, col) -> HeatmapCell
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Enhanced sector heatmap
# ---------------------------------------------------------------------------
def generate_sector_heatmap_enhanced(
    timeframes: Optional[list[str]] = None,
) -> str:
    """Generate an enhanced multi-timeframe sector heatmap.

    Args:
        timeframes: list of timeframe codes (1D, 1W, 1M, 3M). Defaults to all.

    Returns:
        ANSI-formatted heatmap string.
    """
    if timeframes is None:
        timeframes = ["1D", "1W", "1M", "3M"]

    sectors = list(SECTOR_ETFS.keys())
    tickers = list(SECTOR_ETFS.values())
    inv = {v: k for k, v in SECTOR_ETFS.items()}

    # Fetch enough data for the longest timeframe
    max_days = max(TIMEFRAME_DAYS.get(tf, 5) for tf in timeframes) + 30
    start = (pd.Timestamp.now() - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")

    try:
        prices = get_adj_close(tickers, start=start)
    except Exception:
        return "[Sector heatmap unavailable — data fetch failed]"

    if prices.empty:
        return "[Sector heatmap unavailable — no data]"

    lines: list[str] = []
    W = 80
    lines.append(f"\n{BOLD}{CYAN}{'=' * W}{RESET}")
    lines.append(f"{BOLD}{WHITE}  METADRON CAPITAL — ENHANCED SECTOR HEATMAP{RESET}")
    lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")
    lines.append(f"  {DIM}Generated: {datetime.now().isoformat()}{RESET}\n")

    # Header row
    header = f"  {'Sector':<28}"
    for tf in timeframes:
        header += f" {tf:>8}"
    header += "   Visual"
    lines.append(f"{BOLD}{header}{RESET}")
    lines.append(f"  {'-' * (28 + 9 * len(timeframes) + 10)}")

    # Compute returns per sector per timeframe
    for col in prices.columns:
        sector = inv.get(col, col)
        r = prices[col].dropna()
        if len(r) < 2:
            continue

        row = f"  {sector:<28}"
        visual = ""
        for tf in timeframes:
            days = TIMEFRAME_DAYS.get(tf, 2)
            if len(r) >= days + 1:
                ret = float(r.iloc[-1] / r.iloc[-days - 1] - 1) if r.iloc[-days - 1] != 0 else 0.0
            elif len(r) >= 2:
                ret = float(r.iloc[-1] / r.iloc[0] - 1) if r.iloc[0] != 0 else 0.0
            else:
                ret = 0.0

            color = _return_to_color(ret)
            row += f" {color}{ret:>+7.2%}{RESET}"
            visual += _return_to_block(ret)

        row += f"  {visual}"
        lines.append(row)

    lines.append(f"\n  {DIM}Legend: {_GREEN_GRADIENT[-1]}██{RESET}{DIM} Strong Up  "
                 f"{_GREEN_GRADIENT[1]}▓▓{RESET}{DIM} Up  "
                 f"{_NEUTRAL_COLOR}░░{RESET}{DIM} Flat  "
                 f"{_RED_GRADIENT[1]}▒▒{RESET}{DIM} Down  "
                 f"{_RED_GRADIENT[-1]}██{RESET}{DIM} Strong Down{RESET}")
    lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factor heatmap
# ---------------------------------------------------------------------------
def generate_factor_heatmap(
    timeframes: Optional[list[str]] = None,
) -> str:
    """Generate a factor performance heatmap.

    Covers: Momentum, Value, Quality, Size, Low Vol, and optionally
    additional factor ETFs.
    """
    if timeframes is None:
        timeframes = ["1D", "1W", "1M", "3M"]

    factor_map = {
        "Momentum": "MTUM",
        "Value": "VLUE",
        "Quality": "QUAL",
        "Size": "SIZE",
        "Low Volatility": "USMV",
    }

    tickers = list(factor_map.values())
    inv = {v: k for k, v in factor_map.items()}

    max_days = max(TIMEFRAME_DAYS.get(tf, 5) for tf in timeframes) + 30
    start = (pd.Timestamp.now() - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")

    try:
        prices = get_adj_close(tickers, start=start)
    except Exception:
        return "[Factor heatmap unavailable — data fetch failed]"

    if prices.empty:
        return "[Factor heatmap unavailable — no data]"

    lines: list[str] = []
    W = 72
    lines.append(f"\n{BOLD}{MAGENTA}{'=' * W}{RESET}")
    lines.append(f"{BOLD}{WHITE}  FACTOR HEATMAP{RESET}")
    lines.append(f"{BOLD}{MAGENTA}{'=' * W}{RESET}")

    header = f"  {'Factor':<22}"
    for tf in timeframes:
        header += f" {tf:>8}"
    header += "   Visual"
    lines.append(f"{BOLD}{header}{RESET}")
    lines.append(f"  {'-' * (22 + 9 * len(timeframes) + 10)}")

    for col in prices.columns:
        factor = inv.get(col, col)
        r = prices[col].dropna()
        if len(r) < 2:
            continue

        row = f"  {factor:<22}"
        visual = ""
        for tf in timeframes:
            days = TIMEFRAME_DAYS.get(tf, 2)
            if len(r) >= days + 1:
                ret = float(r.iloc[-1] / r.iloc[-days - 1] - 1) if r.iloc[-days - 1] != 0 else 0.0
            elif len(r) >= 2:
                ret = float(r.iloc[-1] / r.iloc[0] - 1) if r.iloc[0] != 0 else 0.0
            else:
                ret = 0.0

            color = _return_to_color(ret)
            row += f" {color}{ret:>+7.2%}{RESET}"
            visual += _return_to_block(ret)

        row += f"  {visual}"
        lines.append(row)

    lines.append(f"{BOLD}{MAGENTA}{'=' * W}{RESET}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
def generate_correlation_heatmap(
    lookback_days: int = 60,
    tickers: Optional[list[str]] = None,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Generate a correlation heatmap between assets.

    By default uses sector ETFs. Pass custom tickers/labels for other views.

    Args:
        lookback_days: number of trading days for correlation
        tickers: list of tickers; defaults to sector ETFs
        labels: ticker -> display label mapping

    Returns:
        ANSI correlation heatmap string.
    """
    if tickers is None:
        tickers = list(SECTOR_ETFS.values())
        labels = {v: k[:8] for k, v in SECTOR_ETFS.items()}
    elif labels is None:
        labels = {t: t for t in tickers}

    start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

    try:
        rets = get_returns(tickers, start=start)
    except Exception:
        return "[Correlation heatmap unavailable — data fetch failed]"

    if rets.empty or len(rets) < 10:
        return "[Correlation heatmap unavailable — insufficient data]"

    # Compute correlation matrix
    corr = rets.corr()

    lines: list[str] = []
    W = 10 + len(tickers) * 9
    lines.append(f"\n{BOLD}{BLUE}{'=' * max(W, 60)}{RESET}")
    lines.append(f"{BOLD}{WHITE}  CORRELATION HEATMAP ({lookback_days}D lookback){RESET}")
    lines.append(f"{BOLD}{BLUE}{'=' * max(W, 60)}{RESET}")

    # Header
    header = f"  {'':>10}"
    for t in tickers:
        lbl = labels.get(t, t)[:7]
        header += f" {lbl:>7}"
    lines.append(f"{DIM}{header}{RESET}")
    lines.append(f"  {'-' * (10 + len(tickers) * 8)}")

    for i, t1 in enumerate(tickers):
        lbl1 = labels.get(t1, t1)[:10]
        row = f"  {lbl1:>10}"
        for j, t2 in enumerate(tickers):
            if t1 in corr.index and t2 in corr.columns:
                c = float(corr.loc[t1, t2])
            else:
                c = 0.0
            color = _corr_to_color(c)
            if i == j:
                row += f" {DIM}  1.00{RESET}"
            else:
                row += f" {color}{c:>+6.2f}{RESET}"
        lines.append(row)

    lines.append(f"\n  {DIM}Legend: "
                 f"\033[38;5;46m+0.7+{RESET} "
                 f"\033[38;5;34m+0.4{RESET} "
                 f"{_NEUTRAL_COLOR} 0.0{RESET} "
                 f"\033[38;5;160m-0.4{RESET} "
                 f"\033[38;5;196m-0.7-{RESET}")
    lines.append(f"{BOLD}{BLUE}{'=' * max(W, 60)}{RESET}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio exposure heatmap
# ---------------------------------------------------------------------------
def generate_exposure_heatmap(
    positions: dict[str, dict],
    nav: float = 1_000_000.0,
) -> str:
    """Generate a portfolio exposure heatmap by sector and size.

    Args:
        positions: dict ticker -> {quantity, current_price, sector, ...}
        nav: net asset value

    Returns:
        ANSI heatmap string showing portfolio concentration.
    """
    if not positions:
        return "[Exposure heatmap: no positions]"

    # Group by sector
    sector_data: dict[str, dict] = {}
    for ticker, pos in positions.items():
        sector = pos.get("sector", "Unknown")
        qty = abs(int(pos.get("quantity", pos.get("qty", 0))))
        price = float(pos.get("current_price", pos.get("price", 0)))
        mv = qty * price
        weight = mv / nav if nav > 0 else 0.0

        if sector not in sector_data:
            sector_data[sector] = {"total_weight": 0.0, "positions": []}
        sector_data[sector]["total_weight"] += weight
        sector_data[sector]["positions"].append({
            "ticker": ticker,
            "weight": weight,
            "market_value": mv,
        })

    # Sort by total weight
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]["total_weight"], reverse=True)

    lines: list[str] = []
    W = 72
    lines.append(f"\n{BOLD}{GREEN}{'=' * W}{RESET}")
    lines.append(f"{BOLD}{WHITE}  PORTFOLIO EXPOSURE HEATMAP{RESET}")
    lines.append(f"{BOLD}{GREEN}{'=' * W}{RESET}")
    lines.append(f"  NAV: ${nav:,.0f}\n")

    # Sector bars
    max_weight = max(d["total_weight"] for _, d in sorted_sectors) if sorted_sectors else 0.01
    bar_max = 40

    for sector, data in sorted_sectors:
        w = data["total_weight"]
        bar_len = int(w / max_weight * bar_max) if max_weight > 0 else 0

        # Color based on concentration
        if w > 0.20:
            color = "\033[38;5;196m"  # red — overconcentrated
        elif w > 0.12:
            color = "\033[38;5;208m"  # orange — elevated
        elif w > 0.05:
            color = "\033[38;5;46m"   # green — normal
        else:
            color = "\033[38;5;240m"  # gray — small

        bar = color + BLOCK_FULL * (bar_len // 2) + (BLOCK_LIGHT if bar_len % 2 else "") + RESET
        lines.append(f"  {sector:<25} {w:>6.1%} {bar}")

        # Individual positions within sector
        for pos in sorted(data["positions"], key=lambda p: p["weight"], reverse=True):
            pw = pos["weight"]
            if pw > 0.005:  # Only show meaningful positions
                mini_bar_len = int(pw / max_weight * bar_max) if max_weight > 0 else 0
                mini_bar = DIM + "·" * mini_bar_len + RESET
                lines.append(f"    {pos['ticker']:<21} {pw:>6.2%} {mini_bar}")

    # Cash allocation
    cash_weight = 1.0 - sum(d["total_weight"] for _, d in sorted_sectors)
    if cash_weight > 0.01:
        lines.append(f"\n  {'Cash':<25} {cash_weight:>6.1%} {DIM}{'░' * int(cash_weight * bar_max)}{RESET}")

    # Concentration metrics
    weights = [d["total_weight"] for _, d in sorted_sectors]
    hhi = sum(w ** 2 for w in weights) if weights else 0.0
    top3 = sum(sorted(weights, reverse=True)[:3]) if len(weights) >= 3 else sum(weights)
    lines.append(f"\n  {DIM}HHI: {hhi:.4f}  |  Top 3 sectors: {top3:.1%}  |  "
                 f"Sectors: {len(sorted_sectors)}{RESET}")
    lines.append(f"{BOLD}{GREEN}{'=' * W}{RESET}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# International market heatmap
# ---------------------------------------------------------------------------
def generate_international_heatmap(
    timeframes: Optional[list[str]] = None,
) -> str:
    """Generate an international market heatmap.

    Shows US, Europe, Asia, and Emerging market ETF performance
    across multiple timeframes.
    """
    if timeframes is None:
        timeframes = ["1D", "1W", "1M", "3M"]

    max_days = max(TIMEFRAME_DAYS.get(tf, 5) for tf in timeframes) + 30
    start = (pd.Timestamp.now() - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")

    # Collect all tickers
    all_tickers = []
    ticker_to_region: dict[str, str] = {}
    ticker_to_label: dict[str, str] = {}
    for region, markets in INTERNATIONAL_MARKETS.items():
        for label, ticker in markets.items():
            all_tickers.append(ticker)
            ticker_to_region[ticker] = region
            ticker_to_label[ticker] = label

    try:
        prices = get_adj_close(all_tickers, start=start)
    except Exception:
        return "[International heatmap unavailable — data fetch failed]"

    if prices.empty:
        return "[International heatmap unavailable — no data]"

    lines: list[str] = []
    W = 80
    lines.append(f"\n{BOLD}{YELLOW}{'=' * W}{RESET}")
    lines.append(f"{BOLD}{WHITE}  INTERNATIONAL MARKET HEATMAP{RESET}")
    lines.append(f"{BOLD}{YELLOW}{'=' * W}{RESET}")

    header = f"  {'Market':<22} {'Region':<10}"
    for tf in timeframes:
        header += f" {tf:>8}"
    header += "   Visual"
    lines.append(f"{BOLD}{header}{RESET}")
    lines.append(f"  {'-' * (32 + 9 * len(timeframes) + 10)}")

    current_region = None
    for region, markets in INTERNATIONAL_MARKETS.items():
        if region != current_region:
            if current_region is not None:
                lines.append("")
            current_region = region

        for label, ticker in markets.items():
            if ticker not in prices.columns:
                continue
            r = prices[ticker].dropna()
            if len(r) < 2:
                continue

            row = f"  {label:<22} {DIM}{region:<10}{RESET}"
            visual = ""
            for tf in timeframes:
                days = TIMEFRAME_DAYS.get(tf, 2)
                if len(r) >= days + 1:
                    ret = float(r.iloc[-1] / r.iloc[-days - 1] - 1) if r.iloc[-days - 1] != 0 else 0.0
                elif len(r) >= 2:
                    ret = float(r.iloc[-1] / r.iloc[0] - 1) if r.iloc[0] != 0 else 0.0
                else:
                    ret = 0.0

                color = _return_to_color(ret)
                row += f" {color}{ret:>+7.2%}{RESET}"
                visual += _return_to_block(ret)

            row += f"  {visual}"
            lines.append(row)

    lines.append(f"\n{BOLD}{YELLOW}{'=' * W}{RESET}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Asset class heatmap
# ---------------------------------------------------------------------------
def generate_asset_class_heatmap(
    timeframes: Optional[list[str]] = None,
) -> str:
    """Generate a heatmap across all asset classes.

    Covers: Sectors, Factors, Fixed Income, Commodities, Volatility.
    """
    if timeframes is None:
        timeframes = ["1D", "1W", "1M"]

    asset_classes = {
        "Fixed Income": {
            "20Y Treasury": "TLT",
            "7-10Y Treasury": "IEF",
            "1-3Y Treasury": "SHY",
            "IG Corporate": "LQD",
            "HY Corporate": "HYG",
            "TIPS": "TIP",
            "Agg Bond": "AGG",
        },
        "Commodities": {
            "Gold": "GLD",
            "Silver": "SLV",
            "Crude Oil": "USO",
            "Agriculture": "DBA",
            "Broad Commodity": "DBC",
        },
        "Volatility": {
            "VIX Short-Term": "VXX",
            "Short VIX": "SVXY",
            "Ultra VIX": "UVXY",
        },
    }

    max_days = max(TIMEFRAME_DAYS.get(tf, 5) for tf in timeframes) + 30
    start = (pd.Timestamp.now() - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")

    all_tickers = []
    ticker_label = {}
    ticker_class = {}
    for cls, assets in asset_classes.items():
        for label, ticker in assets.items():
            all_tickers.append(ticker)
            ticker_label[ticker] = label
            ticker_class[ticker] = cls

    try:
        prices = get_adj_close(all_tickers, start=start)
    except Exception:
        return "[Asset class heatmap unavailable — data fetch failed]"

    if prices.empty:
        return "[Asset class heatmap unavailable — no data]"

    lines: list[str] = []
    W = 72
    lines.append(f"\n{BOLD}{CYAN}{'=' * W}{RESET}")
    lines.append(f"{BOLD}{WHITE}  ASSET CLASS HEATMAP{RESET}")
    lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")

    header = f"  {'Asset':<22}"
    for tf in timeframes:
        header += f" {tf:>8}"
    header += "   Visual"
    lines.append(f"{BOLD}{header}{RESET}")
    lines.append(f"  {'-' * (22 + 9 * len(timeframes) + 10)}")

    current_class = None
    for cls, assets in asset_classes.items():
        if cls != current_class:
            if current_class is not None:
                lines.append("")
            lines.append(f"  {BOLD}{YELLOW}{cls}{RESET}")
            current_class = cls

        for label, ticker in assets.items():
            if ticker not in prices.columns:
                continue
            r = prices[ticker].dropna()
            if len(r) < 2:
                continue

            row = f"    {label:<20}"
            visual = ""
            for tf in timeframes:
                days = TIMEFRAME_DAYS.get(tf, 2)
                if len(r) >= days + 1:
                    ret = float(r.iloc[-1] / r.iloc[-days - 1] - 1) if r.iloc[-days - 1] != 0 else 0.0
                elif len(r) >= 2:
                    ret = float(r.iloc[-1] / r.iloc[0] - 1) if r.iloc[0] != 0 else 0.0
                else:
                    ret = 0.0

                color = _return_to_color(ret)
                row += f" {color}{ret:>+7.2%}{RESET}"
                visual += _return_to_block(ret)

            row += f"  {visual}"
            lines.append(row)

    lines.append(f"\n{BOLD}{CYAN}{'=' * W}{RESET}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified dashboard
# ---------------------------------------------------------------------------
def generate_full_heatmap_dashboard(
    positions: Optional[dict] = None,
    nav: float = 1_000_000.0,
    timeframes: Optional[list[str]] = None,
    include_correlation: bool = True,
    include_international: bool = True,
    include_asset_class: bool = True,
) -> str:
    """Generate the complete heatmap dashboard combining all views.

    Args:
        positions: portfolio positions for exposure heatmap
        nav: net asset value
        timeframes: list of timeframe codes
        include_correlation: whether to include correlation matrix
        include_international: include intl markets
        include_asset_class: include asset classes

    Returns:
        Combined ANSI string with all heatmaps.
    """
    if timeframes is None:
        timeframes = ["1D", "1W", "1M", "3M"]

    sections: list[str] = []

    # 1. Enhanced sector heatmap
    try:
        sections.append(generate_sector_heatmap_enhanced(timeframes))
    except Exception as e:
        sections.append(f"[Sector heatmap error: {e}]")

    # 2. Factor heatmap
    try:
        sections.append(generate_factor_heatmap(timeframes))
    except Exception as e:
        sections.append(f"[Factor heatmap error: {e}]")

    # 3. Portfolio exposure
    if positions:
        try:
            sections.append(generate_exposure_heatmap(positions, nav))
        except Exception as e:
            sections.append(f"[Exposure heatmap error: {e}]")

    # 4. Correlation
    if include_correlation:
        try:
            sections.append(generate_correlation_heatmap())
        except Exception as e:
            sections.append(f"[Correlation heatmap error: {e}]")

    # 5. International
    if include_international:
        try:
            sections.append(generate_international_heatmap(timeframes))
        except Exception as e:
            sections.append(f"[International heatmap error: {e}]")

    # 6. Asset classes
    if include_asset_class:
        try:
            sections.append(generate_asset_class_heatmap(timeframes[:3]))
        except Exception as e:
            sections.append(f"[Asset class heatmap error: {e}]")

    return "\n".join(sections)
