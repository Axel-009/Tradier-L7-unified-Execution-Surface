"""MissedOpportunities — Post-close analysis of missed trading opportunities.

Scans the universe for stocks that moved significantly but were not captured.
Categorizes misses by root cause, tracks recurring patterns, computes
opportunity cost, and feeds learnings back to ML models and risk gates.

Analysis types:
    1. MissedOpportunityDetector  — daily scan for >3% movers
    2. Pattern tracking           — recurring miss patterns, sector bias, time-of-day
    3. Root cause analysis        — which gate/model/filter caused the miss
    4. Opportunity cost           — theoretical P&L, risk-adjusted
    5. Learning integration       — feedback to ML, risk gates, conviction override
    6. Reporting                  — daily/weekly summaries, ASCII tables
"""

import json
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from ..data.yahoo_data import get_adj_close, get_returns
except ImportError:
    get_adj_close = None
    get_returns = None

try:
    from ..data.universe_engine import get_engine, SECTOR_ETFS
except ImportError:
    get_engine = None
    SECTOR_ETFS = {}

try:
    from ..signals.macro_engine import MacroSnapshot, MarketRegime
except ImportError:
    MacroSnapshot = None
    MarketRegime = None

try:
    from ..execution.paper_broker import SignalType
except ImportError:
    SignalType = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOVE_THRESHOLD_PCT = 0.03          # 3% single-day move
LARGE_MOVE_THRESHOLD_PCT = 0.05   # 5% = "large" move
EXTREME_MOVE_THRESHOLD_PCT = 0.10  # 10% = "extreme" move
DEFAULT_SLIPPAGE_BPS = 5.0         # 5 bps slippage for theoretical fills
RISK_FREE_RATE = 0.05              # 5% for Sharpe contribution calc
DEFAULT_POSITION_PCT = 0.02        # 2% of NAV per position (normal allocation)
REPORT_LOG_DIR = Path("logs/missed_opportunities")
MAX_REPORT_HISTORY = 90            # Days of history to retain


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class MissType(str, Enum):
    """Classification of how the opportunity was missed."""
    SIGNAL_NOT_GENERATED = "SIGNAL_NOT_GENERATED"
    SIGNAL_GENERATED_NOT_EXECUTED = "SIGNAL_GENERATED_NOT_EXECUTED"
    RISK_GATE_REJECTION = "RISK_GATE_REJECTION"
    LIQUIDITY_FILTER = "LIQUIDITY_FILTER"
    CORRELATION_FILTER = "CORRELATION_FILTER"
    ML_BLIND_SPOT = "ML_BLIND_SPOT"
    NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
    POSITION_LIMIT = "POSITION_LIMIT"


class MoveCategory(str, Enum):
    """Magnitude of the move."""
    NORMAL = "NORMAL"       # 3-5%
    LARGE = "LARGE"         # 5-10%
    EXTREME = "EXTREME"     # 10%+


class TimeOfDay(str, Enum):
    """When the bulk of the move occurred."""
    MORNING = "MORNING"     # First 2 hours
    MIDDAY = "MIDDAY"       # Middle session
    AFTERNOON = "AFTERNOON" # Last 2 hours
    FULL_DAY = "FULL_DAY"   # Distributed across session


class MoveDirection(str, Enum):
    """Direction of the move."""
    UP = "UP"
    DOWN = "DOWN"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MissedOpportunity:
    """A single missed trading opportunity."""
    ticker: str = ""
    date: str = ""
    direction: MoveDirection = MoveDirection.UP
    move_pct: float = 0.0
    move_category: MoveCategory = MoveCategory.NORMAL
    open_price: float = 0.0
    close_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    volume: float = 0.0
    avg_volume: float = 0.0
    sector: str = ""
    miss_type: MissType = MissType.SIGNAL_NOT_GENERATED
    time_of_day: TimeOfDay = TimeOfDay.FULL_DAY
    pre_earnings: bool = False
    theoretical_pnl: float = 0.0
    risk_adjusted_cost: float = 0.0
    position_size_usd: float = 0.0
    slippage_cost: float = 0.0
    signal_generated: bool = False
    signal_type: str = ""
    signal_strength: float = 0.0
    rejection_reason: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "direction": self.direction.value,
            "move_pct": round(self.move_pct, 4),
            "move_category": self.move_category.value,
            "open_price": round(self.open_price, 4),
            "close_price": round(self.close_price, 4),
            "high_price": round(self.high_price, 4),
            "low_price": round(self.low_price, 4),
            "volume": self.volume,
            "sector": self.sector,
            "miss_type": self.miss_type.value,
            "time_of_day": self.time_of_day.value,
            "pre_earnings": self.pre_earnings,
            "theoretical_pnl": round(self.theoretical_pnl, 2),
            "risk_adjusted_cost": round(self.risk_adjusted_cost, 4),
            "position_size_usd": round(self.position_size_usd, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "signal_generated": self.signal_generated,
            "signal_type": self.signal_type,
            "signal_strength": round(self.signal_strength, 4),
            "rejection_reason": self.rejection_reason,
            "detail": self.detail,
        }


@dataclass
class PatternSummary:
    """Recurring pattern analysis for missed opportunities."""
    pattern_name: str = ""
    occurrence_count: int = 0
    total_missed_pnl: float = 0.0
    avg_missed_pnl: float = 0.0
    sectors_affected: list = field(default_factory=list)
    miss_types: dict = field(default_factory=dict)
    direction_bias: str = ""
    time_of_day_bias: str = ""
    first_seen: str = ""
    last_seen: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "occurrence_count": self.occurrence_count,
            "total_missed_pnl": round(self.total_missed_pnl, 2),
            "avg_missed_pnl": round(self.avg_missed_pnl, 2),
            "sectors_affected": self.sectors_affected,
            "miss_types": self.miss_types,
            "direction_bias": self.direction_bias,
            "time_of_day_bias": self.time_of_day_bias,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class RootCauseBreakdown:
    """Root cause analysis: why opportunities were missed."""
    risk_gate_rejections: int = 0
    risk_gate_pnl_missed: float = 0.0
    ml_blind_spots: int = 0
    ml_blind_spot_pnl_missed: float = 0.0
    liquidity_filter_false_neg: int = 0
    liquidity_filter_pnl_missed: float = 0.0
    correlation_filter_over_restrict: int = 0
    correlation_filter_pnl_missed: float = 0.0
    signal_not_generated: int = 0
    signal_not_generated_pnl: float = 0.0
    not_in_universe: int = 0
    not_in_universe_pnl: float = 0.0
    position_limit_hits: int = 0
    position_limit_pnl: float = 0.0
    total_missed: int = 0
    total_missed_pnl: float = 0.0

    def add_miss(self, miss: MissedOpportunity):
        """Add a missed opportunity to the root cause breakdown."""
        self.total_missed += 1
        self.total_missed_pnl += miss.theoretical_pnl

        if miss.miss_type == MissType.RISK_GATE_REJECTION:
            self.risk_gate_rejections += 1
            self.risk_gate_pnl_missed += miss.theoretical_pnl
        elif miss.miss_type == MissType.ML_BLIND_SPOT:
            self.ml_blind_spots += 1
            self.ml_blind_spot_pnl_missed += miss.theoretical_pnl
        elif miss.miss_type == MissType.LIQUIDITY_FILTER:
            self.liquidity_filter_false_neg += 1
            self.liquidity_filter_pnl_missed += miss.theoretical_pnl
        elif miss.miss_type == MissType.CORRELATION_FILTER:
            self.correlation_filter_over_restrict += 1
            self.correlation_filter_pnl_missed += miss.theoretical_pnl
        elif miss.miss_type == MissType.SIGNAL_NOT_GENERATED:
            self.signal_not_generated += 1
            self.signal_not_generated_pnl += miss.theoretical_pnl
        elif miss.miss_type == MissType.NOT_IN_UNIVERSE:
            self.not_in_universe += 1
            self.not_in_universe_pnl += miss.theoretical_pnl
        elif miss.miss_type == MissType.POSITION_LIMIT:
            self.position_limit_hits += 1
            self.position_limit_pnl += miss.theoretical_pnl

    def to_dict(self) -> dict:
        return {
            "risk_gate_rejections": self.risk_gate_rejections,
            "risk_gate_pnl_missed": round(self.risk_gate_pnl_missed, 2),
            "ml_blind_spots": self.ml_blind_spots,
            "ml_blind_spot_pnl_missed": round(self.ml_blind_spot_pnl_missed, 2),
            "liquidity_filter_false_neg": self.liquidity_filter_false_neg,
            "liquidity_filter_pnl_missed": round(self.liquidity_filter_pnl_missed, 2),
            "correlation_filter_over_restrict": self.correlation_filter_over_restrict,
            "correlation_filter_pnl_missed": round(self.correlation_filter_pnl_missed, 2),
            "signal_not_generated": self.signal_not_generated,
            "signal_not_generated_pnl": round(self.signal_not_generated_pnl, 2),
            "not_in_universe": self.not_in_universe,
            "not_in_universe_pnl": round(self.not_in_universe_pnl, 2),
            "position_limit_hits": self.position_limit_hits,
            "position_limit_pnl": round(self.position_limit_pnl, 2),
            "total_missed": self.total_missed,
            "total_missed_pnl": round(self.total_missed_pnl, 2),
        }


@dataclass
class SectorBiasAnalysis:
    """Analysis of which sectors we miss most frequently."""
    sector: str = ""
    miss_count: int = 0
    total_missed_pnl: float = 0.0
    avg_move_pct: float = 0.0
    long_misses: int = 0
    short_misses: int = 0
    dominant_miss_type: str = ""
    pct_of_total_misses: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "miss_count": self.miss_count,
            "total_missed_pnl": round(self.total_missed_pnl, 2),
            "avg_move_pct": round(self.avg_move_pct, 4),
            "long_misses": self.long_misses,
            "short_misses": self.short_misses,
            "dominant_miss_type": self.dominant_miss_type,
            "pct_of_total_misses": round(self.pct_of_total_misses, 4),
        }


@dataclass
class LearningFeedback:
    """Feedback payload for ML models and risk gate recalibration."""
    timestamp: str = ""
    ml_blind_spot_tickers: list = field(default_factory=list)
    ml_retrain_priority: float = 0.0
    ml_feature_gaps: list = field(default_factory=list)
    risk_gate_too_conservative: bool = False
    suggested_drawdown_gate_adjust: float = 0.0
    suggested_correlation_cap_adjust: float = 0.0
    sector_weight_adjustments: dict = field(default_factory=dict)
    conviction_gate_adjustments: dict = field(default_factory=dict)
    total_opportunity_cost: float = 0.0
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "ml_blind_spot_tickers": self.ml_blind_spot_tickers,
            "ml_retrain_priority": round(self.ml_retrain_priority, 4),
            "ml_feature_gaps": self.ml_feature_gaps,
            "risk_gate_too_conservative": self.risk_gate_too_conservative,
            "suggested_drawdown_gate_adjust": round(self.suggested_drawdown_gate_adjust, 4),
            "suggested_correlation_cap_adjust": round(self.suggested_correlation_cap_adjust, 4),
            "sector_weight_adjustments": {
                k: round(v, 4) for k, v in self.sector_weight_adjustments.items()
            },
            "conviction_gate_adjustments": self.conviction_gate_adjustments,
            "total_opportunity_cost": round(self.total_opportunity_cost, 2),
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Opportunity Cost Calculator
# ---------------------------------------------------------------------------
class OpportunityCostCalculator:
    """Computes theoretical P&L and risk-adjusted opportunity cost."""

    def __init__(
        self,
        nav: float = 1_000_000.0,
        default_position_pct: float = DEFAULT_POSITION_PCT,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        risk_free_rate: float = RISK_FREE_RATE,
    ):
        self.nav = nav
        self.default_position_pct = default_position_pct
        self.slippage_bps = slippage_bps
        self.risk_free_rate = risk_free_rate

    def calculate(self, miss: MissedOpportunity) -> MissedOpportunity:
        """Fill in theoretical P&L and risk-adjusted cost on a MissedOpportunity."""
        position_usd = self.nav * self.default_position_pct
        miss.position_size_usd = position_usd

        if miss.open_price <= 0:
            return miss

        shares = position_usd / miss.open_price

        # Slippage cost (entry + exit)
        slippage_per_share = miss.open_price * self.slippage_bps / 10_000
        miss.slippage_cost = slippage_per_share * shares * 2

        # Theoretical P&L
        if miss.direction == MoveDirection.UP:
            raw_pnl = (miss.close_price - miss.open_price) * shares
        else:
            raw_pnl = (miss.open_price - miss.close_price) * shares

        miss.theoretical_pnl = raw_pnl - miss.slippage_cost

        # Risk-adjusted opportunity cost (Sharpe contribution)
        if position_usd > 0:
            trade_return = miss.theoretical_pnl / position_usd
            annualized_return = trade_return * 252
            daily_vol = abs(miss.move_pct)
            annualized_vol = daily_vol * math.sqrt(252)
            if annualized_vol > 0:
                sharpe_contribution = (annualized_return - self.risk_free_rate) / annualized_vol
            else:
                sharpe_contribution = 0.0
            miss.risk_adjusted_cost = sharpe_contribution
        else:
            miss.risk_adjusted_cost = 0.0

        return miss

    def calculate_batch(self, misses: list) -> list:
        """Calculate opportunity cost for a batch of missed opportunities."""
        return [self.calculate(m) for m in misses]

    def update_nav(self, new_nav: float):
        """Update NAV for position sizing."""
        self.nav = new_nav


# ---------------------------------------------------------------------------
# Missed Opportunity Detector
# ---------------------------------------------------------------------------
class MissedOpportunityDetector:
    """Scans the universe for stocks that moved >3% in a single day
    and compares against signals that were generated or should have been.
    """

    def __init__(
        self,
        nav: float = 1_000_000.0,
        move_threshold: float = MOVE_THRESHOLD_PCT,
        log_dir: Optional[Path] = None,
    ):
        self.nav = nav
        self.move_threshold = move_threshold
        self.log_dir = log_dir or REPORT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._cost_calc = OpportunityCostCalculator(nav=nav)
        self._history: list[MissedOpportunity] = []
        self._daily_reports: dict[str, list] = {}
        self._sector_map: dict[str, str] = {}

    def _build_sector_map(self):
        """Build ticker -> sector mapping from universe engine."""
        if get_engine is None:
            return
        try:
            engine = get_engine()
            engine.load()
            for sector in engine.get_sectors():
                for sec in engine.get_by_sector(sector):
                    self._sector_map[sec.ticker] = sector
        except Exception:
            pass

    def scan_universe(
        self,
        tickers: Optional[list] = None,
        date: Optional[str] = None,
        executed_signals: Optional[dict] = None,
        generated_signals: Optional[dict] = None,
        rejected_signals: Optional[dict] = None,
    ) -> list:
        """Scan for missed opportunities in the universe.

        Args:
            tickers: List of tickers to scan (defaults to S&P 500 ETF sectors)
            date: Date string to scan (defaults to most recent trading day)
            executed_signals: {ticker: signal_type} that were actually executed
            generated_signals: {ticker: {signal_type, strength, ...}} generated
            rejected_signals: {ticker: {reason, ...}} rejected by risk gates

        Returns:
            List of MissedOpportunity objects
        """
        if not self._sector_map:
            self._build_sector_map()

        executed = executed_signals or {}
        generated = generated_signals or {}
        rejected = rejected_signals or {}

        if tickers is None:
            tickers = list(SECTOR_ETFS.values()) if SECTOR_ETFS else []
            tickers.extend(list(generated.keys()))
            tickers.extend(list(rejected.keys()))
            tickers = list(set(tickers))

        if not tickers:
            return []

        lookback_start = (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        try:
            if get_adj_close is None:
                return []
            prices = get_adj_close(tickers, start=lookback_start)
            if prices.empty:
                return []
        except Exception:
            return []

        misses = []
        date_str = date or datetime.now().strftime("%Y-%m-%d")

        for ticker in tickers:
            if ticker not in prices.columns:
                continue

            ts = prices[ticker].dropna()
            if len(ts) < 2:
                continue

            close_price = float(ts.iloc[-1])
            prev_close = float(ts.iloc[-2])
            if prev_close <= 0:
                continue

            daily_return = (close_price - prev_close) / prev_close

            if abs(daily_return) < self.move_threshold:
                continue

            # Skip if already executed
            if ticker in executed:
                continue

            direction = MoveDirection.UP if daily_return > 0 else MoveDirection.DOWN
            move_category = self._classify_move(abs(daily_return))

            miss_type = self._determine_miss_type(
                ticker, direction, generated, rejected,
            )

            miss = MissedOpportunity(
                ticker=ticker,
                date=date_str,
                direction=direction,
                move_pct=daily_return,
                move_category=move_category,
                open_price=prev_close,
                close_price=close_price,
                high_price=close_price if daily_return > 0 else prev_close,
                low_price=prev_close if daily_return > 0 else close_price,
                sector=self._sector_map.get(ticker, "Unknown"),
                miss_type=miss_type,
                time_of_day=TimeOfDay.FULL_DAY,
            )

            if ticker in generated:
                sig = generated[ticker]
                miss.signal_generated = True
                miss.signal_type = sig.get("signal_type", "")
                miss.signal_strength = sig.get("strength", 0.0)

            if ticker in rejected:
                rej = rejected[ticker]
                miss.rejection_reason = rej.get("reason", "")
                miss.detail = rej.get("detail", "")

            miss.pre_earnings = self._check_pre_earnings(ticker)
            miss = self._cost_calc.calculate(miss)
            misses.append(miss)

        misses.sort(key=lambda m: abs(m.theoretical_pnl), reverse=True)
        self._history.extend(misses)
        self._daily_reports[date_str] = misses
        self._log_misses(misses, date_str)

        return misses

    def _classify_move(self, abs_return: float) -> MoveCategory:
        """Classify the magnitude of a move."""
        if abs_return >= EXTREME_MOVE_THRESHOLD_PCT:
            return MoveCategory.EXTREME
        elif abs_return >= LARGE_MOVE_THRESHOLD_PCT:
            return MoveCategory.LARGE
        else:
            return MoveCategory.NORMAL

    def _determine_miss_type(
        self,
        ticker: str,
        direction: MoveDirection,
        generated: dict,
        rejected: dict,
    ) -> MissType:
        """Determine why the opportunity was missed."""
        if ticker in rejected:
            reason = rejected[ticker].get("reason", "").lower()
            if "risk" in reason or "drawdown" in reason:
                return MissType.RISK_GATE_REJECTION
            elif "liquidity" in reason or "volume" in reason:
                return MissType.LIQUIDITY_FILTER
            elif "correlation" in reason:
                return MissType.CORRELATION_FILTER
            elif "position" in reason or "limit" in reason:
                return MissType.POSITION_LIMIT
            else:
                return MissType.RISK_GATE_REJECTION

        if ticker in generated:
            return MissType.SIGNAL_GENERATED_NOT_EXECUTED

        if ticker not in self._sector_map and get_engine is not None:
            return MissType.NOT_IN_UNIVERSE

        return MissType.ML_BLIND_SPOT

    def _check_pre_earnings(self, ticker: str) -> bool:
        """Check if the move was a pre-earnings drift.

        Heuristic: earnings seasons are Jan, Apr, Jul, Oct. If we are
        within the first 25 days of an earnings month, flag it.
        """
        now = datetime.now()
        earnings_months = {1, 4, 7, 10}
        if now.month in earnings_months and now.day <= 25:
            return True
        return False

    def _log_misses(self, misses: list, date_str: str):
        """Write missed opportunities to JSONL log."""
        log_file = self.log_dir / f"missed_{date_str.replace('-', '')}.jsonl"
        try:
            with open(log_file, "a") as f:
                for m in misses:
                    f.write(json.dumps(m.to_dict(), default=str) + "\n")
        except Exception:
            pass

    def get_history(self, days: int = 30) -> list:
        """Return missed opportunities from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        return [m for m in self._history if m.date >= cutoff_str]


# ---------------------------------------------------------------------------
# Pattern Tracker
# ---------------------------------------------------------------------------
class PatternTracker:
    """Identifies recurring patterns in missed opportunities."""

    def __init__(self):
        self._patterns: dict[str, PatternSummary] = {}

    def analyze_patterns(self, misses: list) -> dict:
        """Analyze a list of missed opportunities for recurring patterns.

        Returns dict of pattern_name -> PatternSummary.
        """
        self._patterns = {}

        if not misses:
            return self._patterns

        # Pattern 1: Momentum breakouts
        momentum_misses = [
            m for m in misses
            if m.move_category in (MoveCategory.LARGE, MoveCategory.EXTREME)
            and m.direction == MoveDirection.UP
        ]
        if len(momentum_misses) >= 2:
            self._add_pattern("momentum_breakout", momentum_misses)

        # Pattern 2: Sell-off captures (missed shorts)
        selloff_misses = [
            m for m in misses
            if m.direction == MoveDirection.DOWN
            and m.move_category != MoveCategory.NORMAL
        ]
        if len(selloff_misses) >= 2:
            self._add_pattern("selloff_capture", selloff_misses)

        # Pattern 3: Pre-earnings drift
        earnings_misses = [m for m in misses if m.pre_earnings]
        if earnings_misses:
            self._add_pattern("pre_earnings_drift", earnings_misses)

        # Pattern 4: Sector-specific blind spots
        sector_counts = defaultdict(list)
        for m in misses:
            if m.sector and m.sector != "Unknown":
                sector_counts[m.sector].append(m)
        for sector, sector_misses in sector_counts.items():
            if len(sector_misses) >= 3:
                safe_name = sector.lower().replace(" ", "_")
                self._add_pattern(f"sector_blind_spot_{safe_name}", sector_misses)

        # Pattern 5: Risk gate over-restriction
        risk_misses = [m for m in misses if m.miss_type == MissType.RISK_GATE_REJECTION]
        if len(risk_misses) >= 2:
            self._add_pattern("risk_gate_overrestriction", risk_misses)

        # Pattern 6: ML model blind spots
        ml_misses = [m for m in misses if m.miss_type == MissType.ML_BLIND_SPOT]
        if len(ml_misses) >= 2:
            self._add_pattern("ml_model_blind_spot", ml_misses)

        # Pattern 7: Correlation filter over-restriction
        corr_misses = [m for m in misses if m.miss_type == MissType.CORRELATION_FILTER]
        if len(corr_misses) >= 2:
            self._add_pattern("correlation_overrestriction", corr_misses)

        return self._patterns

    def _add_pattern(self, name: str, misses: list):
        """Register a pattern from a list of missed opportunities."""
        if not misses:
            return

        pnls = [m.theoretical_pnl for m in misses]
        sectors = list(set(m.sector for m in misses if m.sector))
        miss_type_counts = defaultdict(int)
        direction_counts = defaultdict(int)
        tod_counts = defaultdict(int)

        for m in misses:
            miss_type_counts[m.miss_type.value] += 1
            direction_counts[m.direction.value] += 1
            tod_counts[m.time_of_day.value] += 1

        dates = sorted(m.date for m in misses if m.date)

        pattern = PatternSummary(
            pattern_name=name,
            occurrence_count=len(misses),
            total_missed_pnl=sum(pnls),
            avg_missed_pnl=sum(pnls) / len(pnls) if pnls else 0.0,
            sectors_affected=sectors,
            miss_types=dict(miss_type_counts),
            direction_bias=max(direction_counts, key=direction_counts.get) if direction_counts else "",
            time_of_day_bias=max(tod_counts, key=tod_counts.get) if tod_counts else "",
            first_seen=dates[0] if dates else "",
            last_seen=dates[-1] if dates else "",
        )
        self._patterns[name] = pattern

    def get_patterns(self) -> dict:
        """Return all identified patterns."""
        return dict(self._patterns)

    def get_top_patterns(self, n: int = 5) -> list:
        """Return top N patterns by total missed P&L."""
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: abs(p.total_missed_pnl),
            reverse=True,
        )
        return sorted_patterns[:n]


# ---------------------------------------------------------------------------
# Sector Bias Analyzer
# ---------------------------------------------------------------------------
class SectorBiasAnalyzer:
    """Analyzes which sectors we miss most frequently."""

    def analyze(self, misses: list) -> list:
        """Return SectorBiasAnalysis for each sector with misses."""
        if not misses:
            return []

        sector_groups = defaultdict(list)
        for m in misses:
            sector = m.sector or "Unknown"
            sector_groups[sector].append(m)

        total_misses = len(misses)
        results = []

        for sector, sector_misses in sector_groups.items():
            miss_type_counts = defaultdict(int)
            for m in sector_misses:
                miss_type_counts[m.miss_type.value] += 1

            dominant_type = max(miss_type_counts, key=miss_type_counts.get) if miss_type_counts else ""

            analysis = SectorBiasAnalysis(
                sector=sector,
                miss_count=len(sector_misses),
                total_missed_pnl=sum(m.theoretical_pnl for m in sector_misses),
                avg_move_pct=float(np.mean([abs(m.move_pct) for m in sector_misses])),
                long_misses=sum(1 for m in sector_misses if m.direction == MoveDirection.UP),
                short_misses=sum(1 for m in sector_misses if m.direction == MoveDirection.DOWN),
                dominant_miss_type=dominant_type,
                pct_of_total_misses=len(sector_misses) / total_misses if total_misses > 0 else 0.0,
            )
            results.append(analysis)

        results.sort(key=lambda a: a.miss_count, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Root Cause Analyzer
# ---------------------------------------------------------------------------
class RootCauseAnalyzer:
    """Categorizes missed opportunities by root cause."""

    def analyze(self, misses: list) -> RootCauseBreakdown:
        """Build a root cause breakdown from missed opportunities."""
        breakdown = RootCauseBreakdown()
        for m in misses:
            breakdown.add_miss(m)
        return breakdown

    def get_top_root_causes(self, breakdown: RootCauseBreakdown) -> list:
        """Return root causes sorted by P&L impact."""
        causes = [
            ("Risk Gate Rejections", breakdown.risk_gate_rejections, breakdown.risk_gate_pnl_missed),
            ("ML Blind Spots", breakdown.ml_blind_spots, breakdown.ml_blind_spot_pnl_missed),
            ("Liquidity Filter", breakdown.liquidity_filter_false_neg, breakdown.liquidity_filter_pnl_missed),
            ("Correlation Filter", breakdown.correlation_filter_over_restrict, breakdown.correlation_filter_pnl_missed),
            ("Signal Not Generated", breakdown.signal_not_generated, breakdown.signal_not_generated_pnl),
            ("Not In Universe", breakdown.not_in_universe, breakdown.not_in_universe_pnl),
            ("Position Limit", breakdown.position_limit_hits, breakdown.position_limit_pnl),
        ]
        causes.sort(key=lambda c: abs(c[2]), reverse=True)
        return causes


# ---------------------------------------------------------------------------
# Learning Integration
# ---------------------------------------------------------------------------
class LearningIntegrator:
    """Generates feedback for ML models and risk gates based on misses."""

    def generate_feedback(
        self,
        misses: list,
        root_cause: RootCauseBreakdown,
        sector_bias: list,
        patterns: dict,
    ) -> LearningFeedback:
        """Generate learning feedback from missed opportunity analysis."""
        feedback = LearningFeedback(
            timestamp=datetime.now().isoformat(),
        )

        if not misses:
            feedback.recommendation = "No missed opportunities detected."
            return feedback

        # ML blind spot tickers for retraining priority
        ml_tickers = [m.ticker for m in misses if m.miss_type == MissType.ML_BLIND_SPOT]
        feedback.ml_blind_spot_tickers = list(set(ml_tickers))

        total = len(misses)
        ml_miss_pct = len(ml_tickers) / total if total > 0 else 0.0
        feedback.ml_retrain_priority = ml_miss_pct

        # Identify feature gaps
        ml_misses = [m for m in misses if m.miss_type == MissType.ML_BLIND_SPOT]
        ml_sectors = defaultdict(int)
        ml_directions = defaultdict(int)
        for m in ml_misses:
            ml_sectors[m.sector] += 1
            ml_directions[m.direction.value] += 1
        if ml_sectors:
            top_sector = max(ml_sectors, key=ml_sectors.get)
            feedback.ml_feature_gaps.append(f"Sector blind spot: {top_sector}")
        if ml_directions:
            top_dir = max(ml_directions, key=ml_directions.get)
            feedback.ml_feature_gaps.append(f"Direction bias: misses {top_dir} moves")

        # Risk gate adjustment suggestions
        risk_pct = root_cause.risk_gate_rejections / total if total > 0 else 0.0
        if risk_pct > 0.30:
            feedback.risk_gate_too_conservative = True
            feedback.suggested_drawdown_gate_adjust = 0.01
            feedback.recommendation += "Risk gates are too conservative (>30% of misses). "

        # Correlation filter adjustment
        corr_pct = root_cause.correlation_filter_over_restrict / total if total > 0 else 0.0
        if corr_pct > 0.20:
            feedback.suggested_correlation_cap_adjust = 0.05
            feedback.recommendation += "Correlation filter over-restricting (>20% of misses). "

        # Sector weight adjustments
        for sa in sector_bias:
            if sa.pct_of_total_misses > 0.25:
                feedback.sector_weight_adjustments[sa.sector] = 0.05

        # Conviction override gate feedback
        if "risk_gate_overrestriction" in patterns:
            p = patterns["risk_gate_overrestriction"]
            feedback.conviction_gate_adjustments["drawdown_gate"] = {
                "current": 0.05,
                "suggested": 0.06,
                "reason": f"Risk gate caused {p.occurrence_count} misses worth ${p.total_missed_pnl:.0f}",
            }

        if "correlation_overrestriction" in patterns:
            p = patterns["correlation_overrestriction"]
            feedback.conviction_gate_adjustments["correlation_cap"] = {
                "current": 0.70,
                "suggested": 0.75,
                "reason": f"Correlation filter caused {p.occurrence_count} misses worth ${p.total_missed_pnl:.0f}",
            }

        feedback.total_opportunity_cost = sum(m.theoretical_pnl for m in misses)

        if not feedback.recommendation:
            feedback.recommendation = (
                f"Total opportunity cost: ${feedback.total_opportunity_cost:,.0f}. "
                f"Review {len(misses)} misses."
            )

        return feedback


# ---------------------------------------------------------------------------
# ASCII Report Generator
# ---------------------------------------------------------------------------
class ReportGenerator:
    """Generates ASCII table reports for missed opportunities."""

    def __init__(self, width: int = 100):
        self.width = width

    def daily_report(
        self,
        misses: list,
        root_cause: RootCauseBreakdown,
        date_str: Optional[str] = None,
    ) -> str:
        """Generate a daily missed opportunity report."""
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        lines = []
        lines.append("=" * self.width)
        lines.append(f"  METADRON CAPITAL — MISSED OPPORTUNITY REPORT  |  {date_str}")
        lines.append("=" * self.width)
        lines.append("")

        total_pnl = sum(m.theoretical_pnl for m in misses)
        long_misses = sum(1 for m in misses if m.direction == MoveDirection.UP)
        short_misses = sum(1 for m in misses if m.direction == MoveDirection.DOWN)

        lines.append(f"  Total Missed Opportunities: {len(misses)}")
        lines.append(f"  Total Opportunity Cost:     ${total_pnl:>12,.2f}")
        lines.append(f"  Missed Longs:               {long_misses}")
        lines.append(f"  Missed Shorts:              {short_misses}")
        lines.append("")

        # Root cause summary
        lines.append("-" * self.width)
        lines.append("  ROOT CAUSE BREAKDOWN")
        lines.append("-" * self.width)
        lines.append(f"  {'Cause':<35} {'Count':>6} {'Missed P&L':>14}")
        lines.append(f"  {'-'*35} {'-'*6} {'-'*14}")
        cause_items = [
            ("Risk Gate Rejections", root_cause.risk_gate_rejections, root_cause.risk_gate_pnl_missed),
            ("ML Blind Spots", root_cause.ml_blind_spots, root_cause.ml_blind_spot_pnl_missed),
            ("Liquidity Filter", root_cause.liquidity_filter_false_neg, root_cause.liquidity_filter_pnl_missed),
            ("Correlation Filter", root_cause.correlation_filter_over_restrict, root_cause.correlation_filter_pnl_missed),
            ("Signal Not Generated", root_cause.signal_not_generated, root_cause.signal_not_generated_pnl),
            ("Not In Universe", root_cause.not_in_universe, root_cause.not_in_universe_pnl),
            ("Position Limit", root_cause.position_limit_hits, root_cause.position_limit_pnl),
        ]
        for name, count, pnl in cause_items:
            if count > 0:
                lines.append(f"  {name:<35} {count:>6} ${pnl:>12,.2f}")
        lines.append("")

        # Top 10
        lines.append("-" * self.width)
        lines.append("  TOP 10 MISSED OPPORTUNITIES (by opportunity cost)")
        lines.append("-" * self.width)
        header = (
            f"  {'Ticker':<8} {'Dir':<6} {'Move%':>7} {'Category':<10} "
            f"{'Sector':<20} {'Theo P&L':>12} {'Miss Type':<28}"
        )
        lines.append(header)
        lines.append(
            f"  {'-'*8} {'-'*6} {'-'*7} {'-'*10} "
            f"{'-'*20} {'-'*12} {'-'*28}"
        )

        for m in misses[:10]:
            dir_str = "LONG" if m.direction == MoveDirection.UP else "SHORT"
            lines.append(
                f"  {m.ticker:<8} {dir_str:<6} {m.move_pct:>+6.2%} "
                f"{m.move_category.value:<10} {m.sector:<20} "
                f"${m.theoretical_pnl:>10,.2f} {m.miss_type.value:<28}"
            )
        lines.append("")
        lines.append("=" * self.width)
        return "\n".join(lines)

    def weekly_summary(
        self,
        weekly_misses: list,
        patterns: dict,
        sector_bias: list,
        feedback: LearningFeedback,
    ) -> str:
        """Generate a weekly summary report."""
        lines = []
        lines.append("=" * self.width)
        lines.append("  METADRON CAPITAL — WEEKLY MISSED OPPORTUNITY SUMMARY")
        lines.append(f"  Week ending: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("=" * self.width)
        lines.append("")

        total_pnl = sum(m.theoretical_pnl for m in weekly_misses)
        lines.append(f"  Total Missed This Week:  {len(weekly_misses)}")
        lines.append(f"  Total Opportunity Cost:  ${total_pnl:>12,.2f}")
        if weekly_misses:
            lines.append(f"  Avg Cost per Miss:       ${total_pnl / len(weekly_misses):>12,.2f}")
        else:
            lines.append(f"  Avg Cost per Miss:       $        0.00")
        lines.append("")

        # Pattern summary
        lines.append("-" * self.width)
        lines.append("  RECURRING PATTERNS")
        lines.append("-" * self.width)
        lines.append(
            f"  {'Pattern':<40} {'Count':>6} {'Total Cost':>14} {'Direction':>10}"
        )
        lines.append(
            f"  {'-'*40} {'-'*6} {'-'*14} {'-'*10}"
        )
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: abs(x[1].total_missed_pnl),
            reverse=True,
        )
        for name, pattern in sorted_patterns:
            lines.append(
                f"  {pattern.pattern_name:<40} {pattern.occurrence_count:>6} "
                f"${pattern.total_missed_pnl:>12,.2f} {pattern.direction_bias:>10}"
            )
        lines.append("")

        # Sector bias
        lines.append("-" * self.width)
        lines.append("  SECTOR BIAS ANALYSIS")
        lines.append("-" * self.width)
        lines.append(
            f"  {'Sector':<25} {'Misses':>7} {'% Total':>8} "
            f"{'Long':>6} {'Short':>6} {'Avg Move':>9}"
        )
        lines.append(
            f"  {'-'*25} {'-'*7} {'-'*8} "
            f"{'-'*6} {'-'*6} {'-'*9}"
        )
        for sa in sector_bias[:10]:
            lines.append(
                f"  {sa.sector:<25} {sa.miss_count:>7} "
                f"{sa.pct_of_total_misses:>7.1%} {sa.long_misses:>6} "
                f"{sa.short_misses:>6} {sa.avg_move_pct:>+8.2%}"
            )
        lines.append("")

        # Recommendations
        lines.append("-" * self.width)
        lines.append("  LEARNING RECOMMENDATIONS")
        lines.append("-" * self.width)
        if feedback.ml_blind_spot_tickers:
            lines.append(f"  ML Retrain Priority: {feedback.ml_retrain_priority:.0%}")
            tickers_str = ", ".join(feedback.ml_blind_spot_tickers[:10])
            lines.append(f"  Blind Spot Tickers:  {tickers_str}")
        for gap in feedback.ml_feature_gaps:
            lines.append(f"  Feature Gap:         {gap}")
        if feedback.risk_gate_too_conservative:
            lines.append(
                f"  Risk Gate:           TOO CONSERVATIVE — suggest relaxing "
                f"drawdown gate by {feedback.suggested_drawdown_gate_adjust:.0%}"
            )
        if feedback.suggested_correlation_cap_adjust > 0:
            lines.append(
                f"  Correlation Cap:     Suggest relaxing by "
                f"{feedback.suggested_correlation_cap_adjust:.2f}"
            )
        for sector, adj in feedback.sector_weight_adjustments.items():
            lines.append(f"  Sector Weight:       {sector} +{adj:.0%}")
        lines.append(f"  Recommendation:      {feedback.recommendation}")
        lines.append("")
        lines.append("=" * self.width)
        return "\n".join(lines)

    def top_n_table(self, misses: list, n: int = 10) -> str:
        """Generate a compact top-N missed opportunities table."""
        lines = []
        lines.append(f"  TOP {n} MISSED OPPORTUNITIES")
        lines.append(
            f"  {'#':<4} {'Ticker':<8} {'Move':>7} {'Theo P&L':>12} "
            f"{'Risk-Adj':>10} {'Sector':<18} {'Miss Type':<25}"
        )
        lines.append(
            f"  {'-'*4} {'-'*8} {'-'*7} {'-'*12} "
            f"{'-'*10} {'-'*18} {'-'*25}"
        )

        for i, m in enumerate(misses[:n], 1):
            lines.append(
                f"  {i:<4} {m.ticker:<8} {m.move_pct:>+6.2%} "
                f"${m.theoretical_pnl:>10,.2f} {m.risk_adjusted_cost:>+9.2f} "
                f"{m.sector:<18} {m.miss_type.value:<25}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MissedOpportunityAnalyzer — Orchestrator
# ---------------------------------------------------------------------------
class MissedOpportunityAnalyzer:
    """Top-level orchestrator for missed opportunity analysis.

    Combines detection, pattern tracking, root cause analysis,
    opportunity cost calculation, and learning feedback.
    """

    def __init__(
        self,
        nav: float = 1_000_000.0,
        move_threshold: float = MOVE_THRESHOLD_PCT,
        log_dir: Optional[Path] = None,
    ):
        self.nav = nav
        self.detector = MissedOpportunityDetector(
            nav=nav,
            move_threshold=move_threshold,
            log_dir=log_dir,
        )
        self.pattern_tracker = PatternTracker()
        self.sector_analyzer = SectorBiasAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.learning_integrator = LearningIntegrator()
        self.report_generator = ReportGenerator()
        self._weekly_misses: list = []

    def run_daily_analysis(
        self,
        tickers: Optional[list] = None,
        executed_signals: Optional[dict] = None,
        generated_signals: Optional[dict] = None,
        rejected_signals: Optional[dict] = None,
    ) -> dict:
        """Run the full daily missed opportunity analysis.

        Returns a dict with misses, root cause, patterns, feedback,
        and report text.
        """
        misses = self.detector.scan_universe(
            tickers=tickers,
            executed_signals=executed_signals,
            generated_signals=generated_signals,
            rejected_signals=rejected_signals,
        )

        self._weekly_misses.extend(misses)

        root_cause = self.root_cause_analyzer.analyze(misses)
        patterns = self.pattern_tracker.analyze_patterns(self._weekly_misses)
        sector_bias = self.sector_analyzer.analyze(misses)

        feedback = self.learning_integrator.generate_feedback(
            misses=misses,
            root_cause=root_cause,
            sector_bias=sector_bias,
            patterns=patterns,
        )

        daily_report = self.report_generator.daily_report(misses, root_cause)
        top_10_table = self.report_generator.top_n_table(misses, n=10)

        self._log_feedback(feedback)

        return {
            "misses": misses,
            "root_cause": root_cause,
            "patterns": patterns,
            "sector_bias": sector_bias,
            "feedback": feedback,
            "daily_report": daily_report,
            "top_10_table": top_10_table,
            "summary": {
                "total_misses": len(misses),
                "total_opportunity_cost": sum(m.theoretical_pnl for m in misses),
                "top_miss_ticker": misses[0].ticker if misses else "",
                "top_miss_pnl": misses[0].theoretical_pnl if misses else 0.0,
            },
        }

    def run_weekly_analysis(self) -> dict:
        """Run weekly summary analysis and reset weekly accumulator."""
        misses = self._weekly_misses
        root_cause = self.root_cause_analyzer.analyze(misses)
        patterns = self.pattern_tracker.analyze_patterns(misses)
        sector_bias = self.sector_analyzer.analyze(misses)
        feedback = self.learning_integrator.generate_feedback(
            misses=misses,
            root_cause=root_cause,
            sector_bias=sector_bias,
            patterns=patterns,
        )

        weekly_report = self.report_generator.weekly_summary(
            weekly_misses=misses,
            patterns=patterns,
            sector_bias=sector_bias,
            feedback=feedback,
        )

        result = {
            "weekly_misses": misses,
            "root_cause": root_cause,
            "patterns": patterns,
            "sector_bias": sector_bias,
            "feedback": feedback,
            "weekly_report": weekly_report,
            "summary": {
                "total_weekly_misses": len(misses),
                "total_weekly_opportunity_cost": sum(m.theoretical_pnl for m in misses),
                "pattern_count": len(patterns),
                "sectors_affected": len(sector_bias),
            },
        }

        self._weekly_misses = []
        return result

    def update_nav(self, new_nav: float):
        """Update NAV for position sizing calculations."""
        self.nav = new_nav
        self.detector.nav = new_nav
        self.detector._cost_calc.update_nav(new_nav)

    def _log_feedback(self, feedback: LearningFeedback):
        """Write learning feedback to the log directory."""
        log_dir = self.detector.log_dir
        feedback_file = log_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            with open(feedback_file, "w") as f:
                f.write(json.dumps(feedback.to_dict(), indent=2, default=str))
        except Exception:
            pass

    def get_all_history(self) -> list:
        """Return all detected missed opportunities."""
        return self.detector.get_history()

    def get_patterns(self) -> dict:
        """Return the latest pattern analysis."""
        return self.pattern_tracker.get_patterns()

    def get_summary_stats(self) -> dict:
        """Return aggregate stats across all history."""
        history = self.detector.get_history()
        if not history:
            return {
                "total_misses": 0,
                "total_opportunity_cost": 0.0,
                "avg_move_pct": 0.0,
                "top_sector": "",
                "top_miss_type": "",
            }

        sector_counts = defaultdict(int)
        miss_type_counts = defaultdict(int)
        for m in history:
            sector_counts[m.sector] += 1
            miss_type_counts[m.miss_type.value] += 1

        return {
            "total_misses": len(history),
            "total_opportunity_cost": sum(m.theoretical_pnl for m in history),
            "avg_move_pct": float(np.mean([abs(m.move_pct) for m in history])),
            "top_sector": max(sector_counts, key=sector_counts.get) if sector_counts else "",
            "top_miss_type": max(miss_type_counts, key=miss_type_counts.get) if miss_type_counts else "",
        }
