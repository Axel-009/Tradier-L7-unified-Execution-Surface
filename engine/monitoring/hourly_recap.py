"""Hourly Recap — intraday monitoring snapshots.

Provides:
    - Hourly portfolio snapshot
    - Intraday P&L tracking
    - Position drift monitoring
    - Sector exposure changes
    - Signal activity log
    - Risk metric updates
    - Volatility regime changes
    - Format: structured dict + ASCII summary
"""

import time
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from collections import deque

import numpy as np
import pandas as pd

from ..data.yahoo_data import get_adj_close, get_returns
from ..data.universe_engine import SECTOR_ETFS


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
WHITE = "\033[97m"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SNAPSHOTS = 24          # keep at most 24 hourly snapshots
MAX_SIGNAL_LOG = 500        # max signals stored per day
DRIFT_THRESHOLD = 0.05      # 5% position drift triggers alert
SECTOR_DRIFT_THRESHOLD = 0.03  # 3% sector drift alert
VOL_REGIME_WINDOW = 20      # rolling window for vol regime


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PositionSnapshot:
    """Single position state at a point in time."""
    ticker: str
    quantity: int = 0
    current_price: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0
    sector: str = ""
    drift_from_target: float = 0.0


@dataclass
class SectorExposure:
    """Sector-level exposure snapshot."""
    sector: str
    weight: float = 0.0
    market_value: float = 0.0
    pnl: float = 0.0
    num_positions: int = 0
    change_from_last: float = 0.0


@dataclass
class SignalLogEntry:
    """Single signal activity log entry."""
    timestamp: str = ""
    signal_type: str = ""
    ticker: str = ""
    direction: str = ""
    confidence: float = 0.0
    source: str = ""
    acted_upon: bool = False
    reason: str = ""


@dataclass
class VolatilityRegimeSnapshot:
    """Volatility regime state at a point in time."""
    timestamp: str = ""
    realized_vol_1d: float = 0.0
    realized_vol_5d: float = 0.0
    realized_vol_20d: float = 0.0
    vix_proxy: float = 0.0
    vol_regime: str = "NORMAL"   # LOW / NORMAL / ELEVATED / EXTREME
    vol_trend: str = "STABLE"    # COMPRESSING / STABLE / EXPANDING
    vol_of_vol: float = 0.0


@dataclass
class RiskUpdate:
    """Incremental risk metric update."""
    timestamp: str = ""
    portfolio_var_95: float = 0.0
    portfolio_var_99: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    max_position_weight: float = 0.0
    concentration_hhi: float = 0.0
    beta_estimate: float = 0.0
    tracking_error_est: float = 0.0


@dataclass
class IntradayPnL:
    """Intraday P&L tracking data."""
    timestamp: str = ""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_change: float = 0.0
    best_performer: str = ""
    best_pnl: float = 0.0
    worst_performer: str = ""
    worst_pnl: float = 0.0
    pnl_by_sector: dict = field(default_factory=dict)


@dataclass
class HourlySnapshot:
    """Complete hourly snapshot of the portfolio state."""
    timestamp: str = ""
    hour: int = 0
    nav: float = 0.0
    cash: float = 0.0
    positions: list = field(default_factory=list)
    sector_exposures: list = field(default_factory=list)
    intraday_pnl: IntradayPnL = field(default_factory=IntradayPnL)
    risk_update: RiskUpdate = field(default_factory=RiskUpdate)
    vol_regime: VolatilityRegimeSnapshot = field(default_factory=VolatilityRegimeSnapshot)
    signal_count: int = 0
    alerts: list = field(default_factory=list)
    drift_alerts: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hourly recap engine
# ---------------------------------------------------------------------------
class HourlyRecapEngine:
    """Tracks hourly portfolio state and produces recap reports.

    Accumulates snapshots throughout the trading day. Provides
    position drift monitoring, sector exposure tracking, signal
    activity logging, and volatility regime detection.
    """

    def __init__(self):
        self._snapshots: deque[HourlySnapshot] = deque(maxlen=MAX_SNAPSHOTS)
        self._signal_log: deque[SignalLogEntry] = deque(maxlen=MAX_SIGNAL_LOG)
        self._target_weights: dict[str, float] = {}
        self._target_sector_weights: dict[str, float] = {}
        self._last_sector_exposures: dict[str, float] = {}
        self._last_pnl: float = 0.0
        self._day_start_nav: Optional[float] = None
        self._vol_history: list[float] = []

    # --- Configuration -------------------------------------------------------

    def set_target_weights(self, weights: dict[str, float]):
        """Set target position weights for drift monitoring.

        Args:
            weights: ticker -> target weight (0-1)
        """
        self._target_weights = dict(weights)

    def set_target_sector_weights(self, weights: dict[str, float]):
        """Set target sector weights for drift monitoring.

        Args:
            weights: sector -> target weight (0-1)
        """
        self._target_sector_weights = dict(weights)

    def set_day_start_nav(self, nav: float):
        """Record the NAV at start of day for intraday P&L tracking."""
        self._day_start_nav = nav
        self._last_pnl = 0.0

    # --- Signal logging ------------------------------------------------------

    def log_signal(
        self,
        signal_type: str,
        ticker: str,
        direction: str,
        confidence: float = 0.0,
        source: str = "",
        acted_upon: bool = False,
        reason: str = "",
    ):
        """Log a signal event for the activity log."""
        entry = SignalLogEntry(
            timestamp=datetime.now().isoformat(),
            signal_type=signal_type,
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            source=source,
            acted_upon=acted_upon,
            reason=reason,
        )
        self._signal_log.append(entry)

    def get_signal_log(self, last_n: int = 50) -> list[dict]:
        """Retrieve recent signal log entries."""
        entries = list(self._signal_log)[-last_n:]
        return [asdict(e) for e in entries]

    # --- Snapshot generation -------------------------------------------------

    def take_snapshot(
        self,
        portfolio_state: dict,
        positions: dict[str, dict],
        recent_returns: Optional[np.ndarray] = None,
    ) -> HourlySnapshot:
        """Take a complete hourly snapshot.

        Args:
            portfolio_state: dict with nav, cash, total_pnl, etc.
            positions: dict of ticker -> {quantity, current_price, avg_cost,
                       unrealized_pnl, sector, ...}
            recent_returns: recent daily portfolio returns for risk calcs

        Returns:
            HourlySnapshot dataclass.
        """
        now = datetime.now()
        snapshot = HourlySnapshot(
            timestamp=now.isoformat(),
            hour=now.hour,
            nav=portfolio_state.get("nav", 0.0),
            cash=portfolio_state.get("cash", 0.0),
        )

        if self._day_start_nav is None:
            self._day_start_nav = snapshot.nav

        # 1. Position snapshots
        nav = max(snapshot.nav, 1.0)
        pos_snapshots = []
        for ticker, pos in positions.items():
            qty = int(pos.get("quantity", pos.get("qty", 0)))
            price = float(pos.get("current_price", pos.get("price", 0)))
            avg_cost = float(pos.get("avg_cost", price))
            mv = qty * price
            weight = mv / nav if nav > 0 else 0.0
            upnl = float(pos.get("unrealized_pnl", (price - avg_cost) * qty))
            pnl_pct = upnl / (avg_cost * abs(qty)) if avg_cost * abs(qty) > 0 else 0.0
            target_w = self._target_weights.get(ticker, 0.0)
            drift = weight - target_w

            ps = PositionSnapshot(
                ticker=ticker,
                quantity=qty,
                current_price=price,
                market_value=mv,
                weight=weight,
                unrealized_pnl=upnl,
                pnl_pct=pnl_pct,
                sector=pos.get("sector", ""),
                drift_from_target=drift,
            )
            pos_snapshots.append(ps)

        pos_snapshots.sort(key=lambda p: abs(p.market_value), reverse=True)
        snapshot.positions = [asdict(p) for p in pos_snapshots]

        # 2. Sector exposures
        sector_map: dict[str, dict] = {}
        for ps in pos_snapshots:
            sector = ps.sector or "Unknown"
            if sector not in sector_map:
                sector_map[sector] = {"weight": 0.0, "mv": 0.0, "pnl": 0.0, "count": 0}
            sector_map[sector]["weight"] += ps.weight
            sector_map[sector]["mv"] += ps.market_value
            sector_map[sector]["pnl"] += ps.unrealized_pnl
            sector_map[sector]["count"] += 1

        sector_exposures = []
        for sector, data in sorted(sector_map.items(), key=lambda x: x[1]["mv"], reverse=True):
            change = data["weight"] - self._last_sector_exposures.get(sector, 0.0)
            se = SectorExposure(
                sector=sector,
                weight=data["weight"],
                market_value=data["mv"],
                pnl=data["pnl"],
                num_positions=data["count"],
                change_from_last=change,
            )
            sector_exposures.append(se)
        snapshot.sector_exposures = [asdict(se) for se in sector_exposures]
        self._last_sector_exposures = {s: d["weight"] for s, d in sector_map.items()}

        # 3. Intraday P&L
        total_pnl = float(portfolio_state.get("total_pnl", 0.0))
        total_unrealized = sum(p.unrealized_pnl for p in pos_snapshots)
        pnl_change = total_pnl + total_unrealized - self._last_pnl

        # Best and worst performers
        best_p = max(pos_snapshots, key=lambda p: p.unrealized_pnl) if pos_snapshots else None
        worst_p = min(pos_snapshots, key=lambda p: p.unrealized_pnl) if pos_snapshots else None

        pnl_by_sector = {}
        for se in sector_exposures:
            pnl_by_sector[se.sector] = se.pnl

        intraday = IntradayPnL(
            timestamp=now.isoformat(),
            total_pnl=total_pnl + total_unrealized,
            realized_pnl=total_pnl,
            unrealized_pnl=total_unrealized,
            pnl_change=pnl_change,
            best_performer=best_p.ticker if best_p else "",
            best_pnl=best_p.unrealized_pnl if best_p else 0.0,
            worst_performer=worst_p.ticker if worst_p else "",
            worst_pnl=worst_p.unrealized_pnl if worst_p else 0.0,
            pnl_by_sector=pnl_by_sector,
        )
        snapshot.intraday_pnl = asdict(intraday)
        self._last_pnl = total_pnl + total_unrealized

        # 4. Risk update
        risk = self._compute_risk_update(pos_snapshots, nav, recent_returns)
        snapshot.risk_update = asdict(risk)

        # 5. Volatility regime
        vol_snap = self._compute_vol_regime(recent_returns)
        snapshot.vol_regime = asdict(vol_snap)

        # 6. Signal count for this hour
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        snapshot.signal_count = sum(
            1 for s in self._signal_log
            if s.timestamp >= hour_start.isoformat()
        )

        # 7. Drift alerts
        drift_alerts = self._check_drift(pos_snapshots, sector_exposures)
        snapshot.drift_alerts = drift_alerts
        snapshot.alerts = drift_alerts

        self._snapshots.append(snapshot)
        return snapshot

    # --- Internal computations -----------------------------------------------

    def _compute_risk_update(
        self,
        positions: list[PositionSnapshot],
        nav: float,
        recent_returns: Optional[np.ndarray],
    ) -> RiskUpdate:
        """Compute incremental risk metrics."""
        risk = RiskUpdate(timestamp=datetime.now().isoformat())

        if not positions:
            return risk

        weights = [p.weight for p in positions]
        long_val = sum(p.market_value for p in positions if p.quantity > 0)
        short_val = sum(abs(p.market_value) for p in positions if p.quantity < 0)

        risk.gross_exposure = (long_val + short_val) / nav if nav > 0 else 0.0
        risk.net_exposure = (long_val - short_val) / nav if nav > 0 else 0.0
        risk.max_position_weight = max(abs(w) for w in weights) if weights else 0.0

        # HHI concentration
        risk.concentration_hhi = sum(w ** 2 for w in weights) if weights else 0.0

        # VaR from recent returns
        if recent_returns is not None and len(recent_returns) >= 10:
            r = np.asarray(recent_returns, dtype=float)
            r = r[np.isfinite(r)]
            if len(r) >= 10:
                sorted_r = np.sort(r)
                idx_95 = max(0, int(0.05 * len(sorted_r)))
                idx_99 = max(0, int(0.01 * len(sorted_r)))
                risk.portfolio_var_95 = float(sorted_r[idx_95])
                risk.portfolio_var_99 = float(sorted_r[idx_99])

                # Simple beta estimate vs SPY
                try:
                    start = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
                    spy_rets = get_returns("SPY", start=start)
                    if isinstance(spy_rets, pd.DataFrame) and not spy_rets.empty:
                        spy_r = spy_rets.iloc[:, 0].values
                        n = min(len(r), len(spy_r))
                        if n >= 10:
                            cov = np.cov(r[-n:], spy_r[-n:])
                            if cov.shape == (2, 2) and cov[1, 1] > 1e-12:
                                risk.beta_estimate = float(cov[0, 1] / cov[1, 1])
                except Exception:
                    pass

        return risk

    def _compute_vol_regime(
        self,
        recent_returns: Optional[np.ndarray],
    ) -> VolatilityRegimeSnapshot:
        """Detect current volatility regime."""
        vol_snap = VolatilityRegimeSnapshot(timestamp=datetime.now().isoformat())

        if recent_returns is None or len(recent_returns) < 5:
            return vol_snap

        r = np.asarray(recent_returns, dtype=float)
        r = r[np.isfinite(r)]

        if len(r) < 5:
            return vol_snap

        # Realized vol at different horizons
        vol_snap.realized_vol_1d = float(np.std(r[-1:])) * np.sqrt(252) if len(r) >= 1 else 0.0
        vol_snap.realized_vol_5d = float(np.std(r[-5:])) * np.sqrt(252) if len(r) >= 5 else 0.0
        vol_snap.realized_vol_20d = float(np.std(r[-20:])) * np.sqrt(252) if len(r) >= 20 else 0.0

        # VIX proxy (20-day realized vol annualized)
        vol_snap.vix_proxy = vol_snap.realized_vol_20d * 100

        # Vol regime classification
        ann_vol = vol_snap.realized_vol_20d
        if ann_vol < 0.10:
            vol_snap.vol_regime = "LOW"
        elif ann_vol < 0.18:
            vol_snap.vol_regime = "NORMAL"
        elif ann_vol < 0.28:
            vol_snap.vol_regime = "ELEVATED"
        else:
            vol_snap.vol_regime = "EXTREME"

        # Vol trend: compare short-term to long-term vol
        if len(r) >= 20:
            vol_short = float(np.std(r[-5:])) * np.sqrt(252)
            vol_long = float(np.std(r[-20:])) * np.sqrt(252)
            ratio = vol_short / vol_long if vol_long > 1e-10 else 1.0
            if ratio < 0.75:
                vol_snap.vol_trend = "COMPRESSING"
            elif ratio > 1.35:
                vol_snap.vol_trend = "EXPANDING"
            else:
                vol_snap.vol_trend = "STABLE"

        # Vol of vol
        if len(r) >= 20:
            rolling_vols = []
            for i in range(5, len(r)):
                window = r[max(0, i - 5):i]
                rolling_vols.append(float(np.std(window)))
            if len(rolling_vols) >= 5:
                vol_snap.vol_of_vol = float(np.std(rolling_vols)) * np.sqrt(252)

        self._vol_history.append(ann_vol)
        return vol_snap

    def _check_drift(
        self,
        positions: list[PositionSnapshot],
        sector_exposures: list[SectorExposure],
    ) -> list[dict]:
        """Check for position and sector drift alerts."""
        alerts = []

        # Position drift
        for pos in positions:
            if abs(pos.drift_from_target) > DRIFT_THRESHOLD and pos.drift_from_target != pos.weight:
                alerts.append({
                    "type": "POSITION_DRIFT",
                    "level": "WARNING",
                    "ticker": pos.ticker,
                    "current_weight": pos.weight,
                    "target_weight": self._target_weights.get(pos.ticker, 0.0),
                    "drift": pos.drift_from_target,
                    "message": (
                        f"{pos.ticker}: weight {pos.weight:.1%} drifted "
                        f"{pos.drift_from_target:+.1%} from target"
                    ),
                })

        # Sector drift
        for se in sector_exposures:
            target_w = self._target_sector_weights.get(se.sector, 0.0)
            if target_w > 0:
                drift = se.weight - target_w
                if abs(drift) > SECTOR_DRIFT_THRESHOLD:
                    alerts.append({
                        "type": "SECTOR_DRIFT",
                        "level": "WARNING",
                        "sector": se.sector,
                        "current_weight": se.weight,
                        "target_weight": target_w,
                        "drift": drift,
                        "message": (
                            f"Sector {se.sector}: weight {se.weight:.1%} drifted "
                            f"{drift:+.1%} from target"
                        ),
                    })

        # Large sector exposure changes
        for se in sector_exposures:
            if abs(se.change_from_last) > SECTOR_DRIFT_THRESHOLD:
                alerts.append({
                    "type": "SECTOR_CHANGE",
                    "level": "INFO",
                    "sector": se.sector,
                    "change": se.change_from_last,
                    "message": (
                        f"Sector {se.sector}: exposure changed "
                        f"{se.change_from_last:+.1%} in last hour"
                    ),
                })

        return alerts

    # --- Retrieval -----------------------------------------------------------

    def get_snapshots(self, last_n: Optional[int] = None) -> list[dict]:
        """Retrieve recent hourly snapshots as dicts."""
        snaps = list(self._snapshots)
        if last_n:
            snaps = snaps[-last_n:]
        return [asdict(s) if not isinstance(s, dict) else s for s in snaps]

    def get_latest_snapshot(self) -> Optional[dict]:
        """Get the most recent snapshot."""
        if self._snapshots:
            s = self._snapshots[-1]
            return asdict(s) if not isinstance(s, dict) else s
        return None

    def get_intraday_pnl_series(self) -> list[dict]:
        """Get intraday P&L as a time series for charting."""
        result = []
        for snap in self._snapshots:
            pnl_data = snap.intraday_pnl if isinstance(snap.intraday_pnl, dict) else asdict(snap.intraday_pnl)
            result.append({
                "timestamp": snap.timestamp,
                "hour": snap.hour,
                "total_pnl": pnl_data.get("total_pnl", 0),
                "realized_pnl": pnl_data.get("realized_pnl", 0),
                "unrealized_pnl": pnl_data.get("unrealized_pnl", 0),
            })
        return result

    def get_vol_regime_history(self) -> list[dict]:
        """Get volatility regime history."""
        result = []
        for snap in self._snapshots:
            vol = snap.vol_regime if isinstance(snap.vol_regime, dict) else asdict(snap.vol_regime)
            result.append(vol)
        return result

    def reset_day(self):
        """Reset for a new trading day."""
        self._snapshots.clear()
        self._signal_log.clear()
        self._last_pnl = 0.0
        self._day_start_nav = None
        self._last_sector_exposures.clear()

    # --- ASCII formatting ----------------------------------------------------

    def format_snapshot(self, snapshot: Optional[HourlySnapshot] = None) -> str:
        """Render an hourly snapshot as an ASCII summary.

        If no snapshot is provided, uses the latest one.
        """
        if snapshot is None:
            if not self._snapshots:
                return "[No hourly snapshots available]"
            snapshot = self._snapshots[-1]

        # Handle both dict and dataclass
        if isinstance(snapshot, dict):
            snap = snapshot
        else:
            snap = asdict(snapshot)

        lines: list[str] = []
        W = 72

        lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")
        lines.append(f"{BOLD}{WHITE}  METADRON CAPITAL — HOURLY RECAP  [{snap.get('timestamp', '')}]{RESET}")
        lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")

        # Portfolio state
        lines.append(f"\n{BOLD}{YELLOW}--- Portfolio State ---{RESET}")
        nav = snap.get("nav", 0)
        cash = snap.get("cash", 0)
        lines.append(f"  NAV:  ${nav:>14,.2f}")
        lines.append(f"  Cash: ${cash:>14,.2f}")

        # Intraday P&L
        pnl = snap.get("intraday_pnl", {})
        if pnl:
            lines.append(f"\n{BOLD}{YELLOW}--- Intraday P&L ---{RESET}")
            total = pnl.get("total_pnl", 0)
            realized = pnl.get("realized_pnl", 0)
            unrealized = pnl.get("unrealized_pnl", 0)
            change = pnl.get("pnl_change", 0)
            color = GREEN if total >= 0 else RED
            lines.append(f"  Total P&L:      {color}${total:>+14,.2f}{RESET}")
            lines.append(f"  Realized:       ${realized:>+14,.2f}")
            lines.append(f"  Unrealized:     ${unrealized:>+14,.2f}")
            lines.append(f"  Hour Change:    {color}${change:>+14,.2f}{RESET}")
            best = pnl.get("best_performer", "")
            worst = pnl.get("worst_performer", "")
            if best:
                lines.append(f"  Best:  {GREEN}{best}{RESET} (${pnl.get('best_pnl', 0):+,.2f})")
            if worst:
                lines.append(f"  Worst: {RED}{worst}{RESET} (${pnl.get('worst_pnl', 0):+,.2f})")

        # Sector exposures
        sector_exps = snap.get("sector_exposures", [])
        if sector_exps:
            lines.append(f"\n{BOLD}{YELLOW}--- Sector Exposure ---{RESET}")
            lines.append(f"  {'Sector':<25} {'Weight':>7} {'P&L':>12} {'Chg':>7} {'Pos':>4}")
            lines.append(f"  {'-' * 55}")
            for se in sector_exps:
                s = se.get("sector", "?")
                w = se.get("weight", 0)
                p = se.get("pnl", 0)
                c = se.get("change_from_last", 0)
                n = se.get("num_positions", 0)
                pc = GREEN if p >= 0 else RED
                cc = GREEN if c > 0 else (RED if c < 0 else DIM)
                lines.append(
                    f"  {s:<25} {w:>6.1%} {pc}${p:>+10,.0f}{RESET} "
                    f"{cc}{c:>+6.1%}{RESET} {n:>4}"
                )

        # Top positions
        positions = snap.get("positions", [])
        if positions:
            lines.append(f"\n{BOLD}{YELLOW}--- Top Positions ---{RESET}")
            lines.append(f"  {'Ticker':<8} {'Qty':>6} {'Price':>10} {'Weight':>7} {'P&L':>12} {'P&L%':>7}")
            lines.append(f"  {'-' * 50}")
            for pos in positions[:10]:
                t = pos.get("ticker", "?")
                q = pos.get("quantity", 0)
                pr = pos.get("current_price", 0)
                w = pos.get("weight", 0)
                pnl_val = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("pnl_pct", 0)
                pc = GREEN if pnl_val >= 0 else RED
                lines.append(
                    f"  {t:<8} {q:>6} ${pr:>9,.2f} {w:>6.1%} "
                    f"{pc}${pnl_val:>+10,.0f}{RESET} {pc}{pnl_pct:>+6.1%}{RESET}"
                )

        # Risk update
        risk = snap.get("risk_update", {})
        if risk:
            lines.append(f"\n{BOLD}{YELLOW}--- Risk Update ---{RESET}")
            lines.append(f"  Gross Exposure:     {risk.get('gross_exposure', 0):>8.2%}")
            lines.append(f"  Net Exposure:       {risk.get('net_exposure', 0):>8.2%}")
            lines.append(f"  Max Position Weight:{risk.get('max_position_weight', 0):>8.2%}")
            lines.append(f"  HHI Concentration:  {risk.get('concentration_hhi', 0):>8.4f}")
            lines.append(f"  Beta Estimate:      {risk.get('beta_estimate', 0):>8.3f}")
            var95 = risk.get("portfolio_var_95", 0)
            if var95 != 0:
                lines.append(f"  VaR (95%):          {RED}{var95:>+7.2%}{RESET}")

        # Volatility regime
        vol = snap.get("vol_regime", {})
        if vol:
            lines.append(f"\n{BOLD}{YELLOW}--- Volatility Regime ---{RESET}")
            regime = vol.get("vol_regime", "UNKNOWN")
            regime_color = GREEN if regime == "LOW" else (
                YELLOW if regime == "NORMAL" else (
                    BRIGHT_RED if regime in ("ELEVATED", "EXTREME") else DIM
                )
            )
            lines.append(f"  Regime:      {regime_color}{regime}{RESET}")
            lines.append(f"  Trend:       {vol.get('vol_trend', 'UNKNOWN')}")
            lines.append(f"  RVol 5D:     {vol.get('realized_vol_5d', 0):>7.2%}")
            lines.append(f"  RVol 20D:    {vol.get('realized_vol_20d', 0):>7.2%}")
            lines.append(f"  VIX Proxy:   {vol.get('vix_proxy', 0):>7.1f}")
            lines.append(f"  Vol-of-Vol:  {vol.get('vol_of_vol', 0):>7.2%}")

        # Signals this hour
        sig_count = snap.get("signal_count", 0)
        lines.append(f"\n{BOLD}{YELLOW}--- Signals This Hour ---{RESET}")
        lines.append(f"  Signal Count: {sig_count}")

        # Drift alerts
        drift_alerts = snap.get("drift_alerts", [])
        if drift_alerts:
            lines.append(f"\n{BOLD}{YELLOW}--- Drift Alerts ---{RESET}")
            for alert in drift_alerts:
                level = alert.get("level", "INFO")
                msg = alert.get("message", "")
                if level == "WARNING":
                    lines.append(f"  {BRIGHT_RED}[{level}]{RESET} {msg}")
                else:
                    lines.append(f"  {DIM}[{level}]{RESET} {msg}")

        lines.append(f"\n{BOLD}{CYAN}{'=' * W}{RESET}")
        return "\n".join(lines)

    def format_day_summary(self) -> str:
        """Render a full day summary from all hourly snapshots."""
        if not self._snapshots:
            return "[No hourly snapshots for today]"

        lines: list[str] = []
        W = 72

        lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")
        lines.append(f"{BOLD}{WHITE}  METADRON CAPITAL — INTRADAY SUMMARY{RESET}")
        lines.append(f"{BOLD}{CYAN}{'=' * W}{RESET}")

        # P&L timeline
        lines.append(f"\n{BOLD}{YELLOW}--- Intraday P&L Timeline ---{RESET}")
        lines.append(f"  {'Hour':>4}  {'NAV':>14}  {'P&L':>12}  {'Chg':>10}")
        lines.append(f"  {'-' * 44}")

        for snap in self._snapshots:
            h = snap.hour if hasattr(snap, "hour") else snap.get("hour", 0)
            n = snap.nav if hasattr(snap, "nav") else snap.get("nav", 0)
            pnl_data = snap.intraday_pnl if hasattr(snap, "intraday_pnl") else snap.get("intraday_pnl", {})
            if isinstance(pnl_data, dict):
                tp = pnl_data.get("total_pnl", 0)
                ch = pnl_data.get("pnl_change", 0)
            else:
                tp = pnl_data.total_pnl
                ch = pnl_data.pnl_change

            pc = GREEN if tp >= 0 else RED
            cc = GREEN if ch >= 0 else RED
            lines.append(
                f"  {h:>4}  ${n:>13,.2f}  {pc}${tp:>+10,.0f}{RESET}  {cc}${ch:>+8,.0f}{RESET}"
            )

        # Vol regime changes
        lines.append(f"\n{BOLD}{YELLOW}--- Vol Regime Changes ---{RESET}")
        prev_regime = None
        for snap in self._snapshots:
            vol = snap.vol_regime if hasattr(snap, "vol_regime") else snap.get("vol_regime", {})
            if isinstance(vol, dict):
                regime = vol.get("vol_regime", "UNKNOWN")
            else:
                regime = vol.vol_regime
            if regime != prev_regime:
                ts = snap.timestamp if hasattr(snap, "timestamp") else snap.get("timestamp", "")
                if prev_regime is not None:
                    lines.append(f"  {ts}  {prev_regime} -> {regime}")
                prev_regime = regime

        if prev_regime:
            lines.append(f"  Current regime: {prev_regime}")
        else:
            lines.append("  No regime changes detected.")

        # All drift alerts
        all_alerts = []
        for snap in self._snapshots:
            alerts = snap.drift_alerts if hasattr(snap, "drift_alerts") else snap.get("drift_alerts", [])
            all_alerts.extend(alerts)

        if all_alerts:
            lines.append(f"\n{BOLD}{YELLOW}--- All Drift Alerts ---{RESET}")
            for alert in all_alerts:
                lines.append(f"  [{alert.get('level', 'INFO')}] {alert.get('message', '')}")

        lines.append(f"\n{BOLD}{CYAN}{'=' * W}{RESET}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------
_engine: Optional[HourlyRecapEngine] = None


def get_hourly_engine() -> HourlyRecapEngine:
    """Get or create the global HourlyRecapEngine singleton."""
    global _engine
    if _engine is None:
        _engine = HourlyRecapEngine()
    return _engine


def take_hourly_snapshot(
    portfolio_state: dict,
    positions: dict[str, dict],
    recent_returns: Optional[np.ndarray] = None,
) -> dict:
    """Convenience: take snapshot and return as dict."""
    engine = get_hourly_engine()
    snap = engine.take_snapshot(portfolio_state, positions, recent_returns)
    return asdict(snap)


def format_latest_recap() -> str:
    """Convenience: format the latest hourly snapshot."""
    engine = get_hourly_engine()
    return engine.format_snapshot()


def export_hourly_csv(
    portfolio_state: dict,
    positions: dict[str, dict],
    output_dir: Optional[str] = None,
) -> str:
    """Export hourly returns snapshot to CSV.

    Appends one row per call to a daily CSV file:
        logs/returns/returns_YYYY-MM-DD.csv

    Columns:
        timestamp, hour, nav, cash, total_pnl, realized_pnl, unrealized_pnl,
        daily_return_pct, gross_exposure, net_exposure, num_positions,
        regime, best_ticker, best_pnl, worst_ticker, worst_pnl

    Returns the path to the CSV file.
    """
    import csv
    from pathlib import Path

    now = datetime.now()
    log_dir = Path(output_dir) if output_dir else Path("logs/returns")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"returns_{now.strftime('%Y-%m-%d')}.csv"

    nav = portfolio_state.get("nav", 0.0)
    cash = portfolio_state.get("cash", 0.0)
    total_pnl = portfolio_state.get("total_pnl", 0.0)
    realized_pnl = portfolio_state.get("realized_pnl", 0.0)
    initial_nav = portfolio_state.get("initial_nav", nav)

    # Compute unrealized from positions
    unrealized = sum(
        float(p.get("unrealized_pnl", 0.0)) for p in positions.values()
    )

    # Daily return %
    daily_return = ((nav - initial_nav) / initial_nav * 100) if initial_nav > 0 else 0.0

    # Exposure
    long_val = sum(
        abs(int(p.get("quantity", 0)) * float(p.get("current_price", 0)))
        for p in positions.values()
        if int(p.get("quantity", 0)) > 0
    )
    short_val = sum(
        abs(int(p.get("quantity", 0)) * float(p.get("current_price", 0)))
        for p in positions.values()
        if int(p.get("quantity", 0)) < 0
    )
    gross_exp = (long_val + short_val) / nav if nav > 0 else 0.0
    net_exp = (long_val - short_val) / nav if nav > 0 else 0.0

    # Best/worst
    best_ticker, best_pnl = "", 0.0
    worst_ticker, worst_pnl = "", 0.0
    for ticker, pos in positions.items():
        upnl = float(pos.get("unrealized_pnl", 0.0))
        if upnl > best_pnl:
            best_ticker, best_pnl = ticker, upnl
        if upnl < worst_pnl:
            worst_ticker, worst_pnl = ticker, upnl

    regime = portfolio_state.get("regime", "UNKNOWN")

    headers = [
        "timestamp", "hour", "nav", "cash", "total_pnl", "realized_pnl",
        "unrealized_pnl", "daily_return_pct", "gross_exposure", "net_exposure",
        "num_positions", "regime", "best_ticker", "best_pnl",
        "worst_ticker", "worst_pnl",
    ]

    row = [
        now.isoformat(), now.hour, f"{nav:.2f}", f"{cash:.2f}",
        f"{total_pnl:.2f}", f"{realized_pnl:.2f}", f"{unrealized:.2f}",
        f"{daily_return:.4f}", f"{gross_exp:.4f}", f"{net_exp:.4f}",
        len(positions), regime, best_ticker, f"{best_pnl:.2f}",
        worst_ticker, f"{worst_pnl:.2f}",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)

    return str(csv_path)
