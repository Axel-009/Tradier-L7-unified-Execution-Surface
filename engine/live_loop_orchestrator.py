"""LiveLoopOrchestrator — End-to-end continuous live loop orchestrator.

Coordinates the entire Metadron Capital Intelligence Platform from data
ingestion through signal generation, intelligence, decision, execution,
learning, and monitoring.  Runs as a continuous 1-minute heartbeat loop
throughout the trading day with cadence-based phase scheduling.

Architecture:
    Phase 1  DATA          (every tick)     Ingestion + Universal Data Pool
    Phase 2  SIGNALS       (1-min cadence)  Macro, Cube, Liquidity, Fundamentals
    Phase 3  INTELLIGENCE  (5-min cadence)  Alpha optimizer, ML ensemble, agents
    Phase 4  DECISION      (on signal Δ)    Decision matrix, beta corridor, options
    Phase 5  EXECUTION     (on approval)    Execution engine, options, futures hedge
    Phase 6  LEARNING      (continuous)     Feedback loops, agent gradients
    Phase 7  MONITORING    (5-min cadence)  P&L, risk, anomaly, snapshot

Daily Schedule:
    08:00-09:30  Pre-market   Full refresh, overnight signals, SEC scan
    09:30        Market open  Full pipeline flush
    09:30-16:00  Intraday     Continuous 1-min heartbeat
    16:00        Market close EOD reconciliation, learning snapshot
    16:00-20:00  After-hours  Reduced frequency, earnings scan
    20:00-08:00  Overnight    Backtesting, ML training, pattern evolution
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# External imports — graceful degradation (rule 8: try/except on ALL)
# ---------------------------------------------------------------------------
try:
    from .signals.macro_engine import MacroEngine, MacroSnapshot, MarketRegime
except ImportError:
    MacroEngine = None
    MacroSnapshot = None
    MarketRegime = None

try:
    from .signals.metadron_cube import MetadronCube, CubeOutput
except ImportError:
    MetadronCube = None
    CubeOutput = None

try:
    from .signals.security_analysis_engine import SecurityAnalysisEngine
except ImportError:
    SecurityAnalysisEngine = None

try:
    from .ml.alpha_optimizer import AlphaOptimizer, AlphaOutput, AlphaSignal
except ImportError:
    AlphaOptimizer = None
    AlphaOutput = None
    AlphaSignal = None

try:
    from .execution.decision_matrix import DecisionMatrix
except ImportError:
    DecisionMatrix = None

try:
    from .execution.execution_engine import ExecutionEngine
except ImportError:
    ExecutionEngine = None

try:
    from .execution.options_engine import OptionsEngine
except ImportError:
    OptionsEngine = None

try:
    from .portfolio.beta_corridor import BetaCorridor, BetaState, BetaAction
except ImportError:
    BetaCorridor = None
    BetaState = None
    BetaAction = None

try:
    from .monitoring.learning_loop import LearningLoop, SignalOutcome, RegimeFeedback
except ImportError:
    LearningLoop = None
    SignalOutcome = None
    RegimeFeedback = None

try:
    from .monitoring.anomaly_detector import AnomalyDetector
except ImportError:
    AnomalyDetector = None

try:
    from .data.universe_engine import UniverseEngine, get_engine
except ImportError:
    UniverseEngine = None
    get_engine = None

try:
    from .agents.sector_bots import SectorBotManager
except ImportError:
    SectorBotManager = None

try:
    from .agents.agent_scorecard import AgentScorecard
except ImportError:
    AgentScorecard = None

try:
    from .monitoring.portfolio_analytics import PortfolioAnalytics
except ImportError:
    PortfolioAnalytics = None

# Components that may not exist yet — future-proofed stubs
try:
    from .data.ingestion_orchestrator import DataIngestionOrchestrator
except ImportError:
    DataIngestionOrchestrator = None

try:
    from .data.universal_pooling import UniversalDataPool
except ImportError:
    UniversalDataPool = None

try:
    from .signals.fed_liquidity_plumbing import FedLiquidityPlumbing
except ImportError:
    FedLiquidityPlumbing = None

try:
    from .ml.qstrader_backtest_bridge import QSTraderBacktestRunner
except ImportError:
    QSTraderBacktestRunner = None

try:
    from intelligence_platform.plugins.gsd_paul_plugin import GSDPlugin, PaulPlugin
except ImportError:
    try:
        from ..intelligence_platform.plugins.gsd_paul_plugin import GSDPlugin, PaulPlugin
    except ImportError:
        GSDPlugin = None
        PaulPlugin = None

try:
    from intelligence_platform.plugins.gsd_workflow_bridge import GSDWorkflowBridge
except ImportError:
    try:
        from ..intelligence_platform.plugins.gsd_workflow_bridge import GSDWorkflowBridge
    except ImportError:
        GSDWorkflowBridge = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class LoopPhase(str, Enum):
    """Phases within a single heartbeat iteration."""
    DATA = "DATA"
    SIGNALS = "SIGNALS"
    INTELLIGENCE = "INTELLIGENCE"
    DECISION = "DECISION"
    EXECUTION = "EXECUTION"
    LEARNING = "LEARNING"
    MONITORING = "MONITORING"


class LoopState(str, Enum):
    """Overall orchestrator state."""
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class MarketSession(str, Enum):
    """Current market session."""
    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    INTRADAY = "INTRADAY"
    MARKET_CLOSE = "MARKET_CLOSE"
    AFTER_HOURS = "AFTER_HOURS"
    OVERNIGHT = "OVERNIGHT"
    WEEKEND = "WEEKEND"


class RiskLevel(str, Enum):
    """Circuit breaker risk level."""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    KILL_SWITCH = "KILL_SWITCH"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    """Result of a single phase execution."""
    phase: str = ""
    success: bool = True
    duration_ms: float = 0.0
    timestamp: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""
    items_processed: int = 0


@dataclass
class HeartbeatResult:
    """Result of a complete heartbeat iteration."""
    iteration: int = 0
    timestamp: str = ""
    total_duration_ms: float = 0.0
    session: str = ""
    phases: Dict[str, PhaseResult] = field(default_factory=dict)
    signals_generated: int = 0
    trades_approved: int = 0
    trades_executed: int = 0
    errors: List[str] = field(default_factory=list)
    risk_level: str = RiskLevel.NORMAL.value


@dataclass
class LoopPerformance:
    """Cumulative loop performance metrics."""
    total_iterations: int = 0
    total_runtime_seconds: float = 0.0
    avg_heartbeat_ms: float = 0.0
    max_heartbeat_ms: float = 0.0
    min_heartbeat_ms: float = 999999.0
    total_signals: int = 0
    total_trades_approved: int = 0
    total_trades_executed: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    max_consecutive_errors: int = 0
    uptime_pct: float = 100.0
    phase_avg_ms: Dict[str, float] = field(default_factory=dict)
    phase_total_ms: Dict[str, float] = field(default_factory=dict)
    phase_call_count: Dict[str, int] = field(default_factory=dict)
    last_heartbeat: str = ""
    started_at: str = ""
    restarts: int = 0


@dataclass
class LoopStatus:
    """Current orchestrator status snapshot."""
    state: str = LoopState.IDLE.value
    session: str = MarketSession.OVERNIGHT.value
    risk_level: str = RiskLevel.NORMAL.value
    current_phase: str = ""
    iteration: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: str = ""
    next_heartbeat_in_seconds: float = 0.0
    components_loaded: Dict[str, bool] = field(default_factory=dict)
    kill_switch_active: bool = False
    performance: Optional[LoopPerformance] = None
    last_macro_regime: str = ""
    last_cube_regime: str = ""
    portfolio_nav: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0


@dataclass
class CircuitBreakerState:
    """Risk circuit breaker state."""
    level: RiskLevel = RiskLevel.NORMAL
    kill_switch_active: bool = False
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    vix_level: float = 20.0
    hy_spread_change_bps: float = 0.0
    breadth_pct: float = 50.0
    last_check: str = ""
    triggers: List[str] = field(default_factory=list)

    # Thresholds
    DRAWDOWN_ELEVATED: float = 0.02
    DRAWDOWN_HIGH: float = 0.05
    DRAWDOWN_CRITICAL: float = 0.08
    DRAWDOWN_KILL: float = 0.12
    VIX_ELEVATED: float = 25.0
    VIX_HIGH: float = 35.0
    VIX_CRITICAL: float = 45.0
    CONSECUTIVE_LOSS_LIMIT: int = 5
    HY_SPREAD_KILL_BPS: float = 35.0
    BREADTH_KILL_PCT: float = 50.0


# ---------------------------------------------------------------------------
# Schedule time boundaries (ET)
# ---------------------------------------------------------------------------
_PRE_MARKET_START = dt_time(8, 0)
_MARKET_OPEN = dt_time(9, 30)
_MARKET_CLOSE = dt_time(16, 0)
_AFTER_HOURS_END = dt_time(20, 0)

# Cadence intervals (seconds)
_HEARTBEAT_INTERVAL = 60          # 1 minute
_SIGNAL_CADENCE = 60              # 1 minute
_INTELLIGENCE_CADENCE = 300       # 5 minutes
_MONITORING_CADENCE = 300         # 5 minutes
_AFTER_HOURS_INTERVAL = 300       # 5 minutes
_OVERNIGHT_INTERVAL = 3600        # 1 hour

# Auto-restart limits
_MAX_CONSECUTIVE_ERRORS = 10
_RESTART_COOLDOWN_SECONDS = 30
_STATE_PERSISTENCE_INTERVAL = 300  # 5 minutes

# Paths
_STATE_DIR = Path("logs/live_loop")
_STATE_FILE = _STATE_DIR / "loop_state.json"
_PERFORMANCE_FILE = _STATE_DIR / "loop_performance.json"


# ---------------------------------------------------------------------------
# LiveLoopOrchestrator
# ---------------------------------------------------------------------------
class LiveLoopOrchestrator:
    """End-to-end continuous live loop orchestrator.

    Coordinates the entire Intelligence Platform from data ingestion
    through execution in a continuous 1-minute heartbeat loop.

    Features:
        - Thread-safe concurrent execution
        - Heartbeat monitoring with timing
        - Auto-restart on failures
        - Risk circuit breaker (KillSwitch)
        - Comprehensive logging
        - State persistence between restarts
        - Graceful shutdown
    """

    def __init__(
        self,
        initial_nav: float = 1_000_000.0,
        broker_type: str = "paper",
        heartbeat_interval: float = _HEARTBEAT_INTERVAL,
        enable_risk_gates: bool = True,
        enable_persistence: bool = True,
        max_consecutive_errors: int = _MAX_CONSECUTIVE_ERRORS,
    ):
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Configuration
        self._initial_nav = initial_nav
        self._broker_type = broker_type
        self._heartbeat_interval = heartbeat_interval
        self._enable_risk_gates = enable_risk_gates
        self._enable_persistence = enable_persistence
        self._max_consecutive_errors = max_consecutive_errors

        # State
        self._state = LoopState.IDLE
        self._session = MarketSession.OVERNIGHT
        self._current_phase = ""
        self._iteration = 0
        self._started_at: Optional[datetime] = None
        self._last_heartbeat_time: Optional[datetime] = None

        # Cadence tracking (last execution timestamps)
        self._last_signal_time: Optional[datetime] = None
        self._last_intelligence_time: Optional[datetime] = None
        self._last_monitoring_time: Optional[datetime] = None
        self._last_state_persist_time: Optional[datetime] = None

        # Results
        self._performance = LoopPerformance()
        self._heartbeat_history: deque[HeartbeatResult] = deque(maxlen=1440)  # 24h of 1-min
        self._circuit_breaker = CircuitBreakerState()

        # Signal state cache (for change detection in decision phase)
        self._last_signals: Dict[str, Any] = {}
        self._pending_trades: List[dict] = []
        self._approved_trades: List[dict] = []

        # Last outputs from each phase
        self._last_macro_snapshot: Any = None
        self._last_cube_output: Any = None
        self._last_alpha_output: Any = None
        self._last_decision_result: Any = None

        # Initialize components
        self._components: Dict[str, Any] = {}
        self._component_status: Dict[str, bool] = {}
        self._init_components()

        # Restore state if available
        if self._enable_persistence:
            self._restore_state()

        logger.info(
            "LiveLoopOrchestrator initialized: nav=%.0f broker=%s interval=%ds components=%d/%d",
            initial_nav, broker_type, heartbeat_interval,
            sum(self._component_status.values()), len(self._component_status),
        )

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------
    def _init_components(self):
        """Initialize all platform components with graceful fallbacks."""
        component_factories: List[Tuple[str, Callable]] = [
            ("data_ingestion", lambda: DataIngestionOrchestrator() if DataIngestionOrchestrator else None),
            ("data_pool", lambda: UniversalDataPool() if UniversalDataPool else None),
            ("universe", lambda: get_engine() if get_engine else None),
            ("fed_liquidity", lambda: FedLiquidityPlumbing() if FedLiquidityPlumbing else None),
            ("macro_engine", lambda: MacroEngine() if MacroEngine else None),
            ("metadron_cube", lambda: MetadronCube() if MetadronCube else None),
            ("security_analysis", lambda: SecurityAnalysisEngine() if SecurityAnalysisEngine else None),
            ("alpha_optimizer", lambda: AlphaOptimizer() if AlphaOptimizer else None),
            ("decision_matrix", lambda: DecisionMatrix() if DecisionMatrix else None),
            ("execution_engine", lambda: ExecutionEngine(
                initial_nav=self._initial_nav,
                broker_type=self._broker_type,
                enable_risk_gates=self._enable_risk_gates,
            ) if ExecutionEngine else None),
            ("options_engine", lambda: OptionsEngine() if OptionsEngine else None),
            ("beta_corridor", lambda: BetaCorridor(nav=self._initial_nav) if BetaCorridor else None),
            ("learning_loop", lambda: LearningLoop() if LearningLoop else None),
            ("anomaly_detector", lambda: AnomalyDetector() if AnomalyDetector else None),
            ("sector_bots", lambda: SectorBotManager() if SectorBotManager else None),
            ("agent_scorecard", lambda: AgentScorecard() if AgentScorecard else None),
            ("portfolio_analytics", lambda: PortfolioAnalytics() if PortfolioAnalytics else None),
            ("gsd_plugin", lambda: GSDPlugin() if GSDPlugin else None),
            ("paul_plugin", lambda: PaulPlugin() if PaulPlugin else None),
            ("paul_orchestrator", lambda: self._init_paul_orchestrator()),
            ("gsd_workflow", lambda: GSDWorkflowBridge() if GSDWorkflowBridge else None),
            ("backtest_runner", lambda: QSTraderBacktestRunner() if QSTraderBacktestRunner else None),
        ]

        for name, factory in component_factories:
            try:
                component = factory()
                self._components[name] = component
                self._component_status[name] = component is not None
                if component is not None:
                    logger.debug("Component loaded: %s", name)
                else:
                    logger.debug("Component unavailable (class not imported): %s", name)
            except Exception as exc:
                self._components[name] = None
                self._component_status[name] = False
                logger.warning("Component init failed: %s — %s", name, exc)

    def _get(self, name: str) -> Any:
        """Retrieve a component by name, returns None if unavailable."""
        return self._components.get(name)

    def _init_paul_orchestrator(self):
        """Initialize the PaulOrchestrator with GSD + Paul + Factory + Enforcement.

        Connects the dynamic agent creation and enforcement system to the
        intelligence platform via the Paul Plugin orchestration layer.
        """
        try:
            from .agents.paul_orchestrator import PaulOrchestrator
            orch = PaulOrchestrator()
            init_status = orch.initialize()
            logger.info("PaulOrchestrator initialized: %s", init_status.get("status"))

            # Attach existing agents if available
            sector_bots = self._get("sector_bots")
            scorecard = self._get("agent_scorecard")

            bot_list = []
            if sector_bots and hasattr(sector_bots, "bots"):
                bot_list = list(sector_bots.bots.values()) if isinstance(sector_bots.bots, dict) else sector_bots.bots

            orch.attach_all_platform_agents(
                sector_bots=bot_list if bot_list else None,
            )
            return orch
        except Exception as exc:
            logger.warning("PaulOrchestrator init failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API — start / stop / status
    # ------------------------------------------------------------------
    def start(self):
        """Begin the continuous live loop in a background thread.

        The loop runs until stop() is called or a fatal error occurs.
        Thread-safe: can be called from any thread.
        """
        with self._lock:
            if self._state == LoopState.RUNNING:
                logger.warning("LiveLoop already running — ignoring start()")
                return

            self._state = LoopState.STARTING
            self._stop_event.clear()
            self._started_at = datetime.now()
            self._performance.started_at = self._started_at.isoformat()
            self._iteration = 0

            logger.info("=" * 70)
            logger.info("METADRON CAPITAL — LIVE LOOP STARTING")
            logger.info("  %s", self._started_at.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("=" * 70)

        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="LiveLoopOrchestrator",
            daemon=True,
        )
        self._loop_thread.start()

    def stop(self):
        """Graceful shutdown of the live loop.

        Signals the loop to stop, waits for the current heartbeat to
        finish, persists state, and joins the thread.
        """
        logger.info("LiveLoop stop requested — initiating graceful shutdown")
        with self._lock:
            self._state = LoopState.STOPPING

        self._stop_event.set()

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=120)
            if self._loop_thread.is_alive():
                logger.error("LiveLoop thread did not exit within 120s timeout")

        with self._lock:
            self._state = LoopState.STOPPED

        # Final state persistence
        if self._enable_persistence:
            self._persist_state()

        logger.info("LiveLoop stopped. Iterations=%d", self._iteration)

    def pause(self):
        """Pause the loop (skips heartbeats until resumed)."""
        with self._lock:
            if self._state == LoopState.RUNNING:
                self._state = LoopState.PAUSED
                logger.info("LiveLoop paused at iteration %d", self._iteration)

    def resume(self):
        """Resume a paused loop."""
        with self._lock:
            if self._state == LoopState.PAUSED:
                self._state = LoopState.RUNNING
                logger.info("LiveLoop resumed at iteration %d", self._iteration)

    # ------------------------------------------------------------------
    # Heartbeat — single loop iteration
    # ------------------------------------------------------------------
    def heartbeat(self) -> HeartbeatResult:
        """Execute a single complete heartbeat iteration.

        Can be called standalone (outside the loop) for testing or
        manual step-through execution.

        Returns:
            HeartbeatResult with timing and outcome for every phase.
        """
        t_start = time.monotonic()
        now = datetime.now()
        self._iteration += 1

        result = HeartbeatResult(
            iteration=self._iteration,
            timestamp=now.isoformat(),
            session=self._session.value,
        )

        logger.info(
            "--- Heartbeat #%d  [%s]  session=%s ---",
            self._iteration, now.strftime("%H:%M:%S"), self._session.value,
        )

        # Determine current session
        self._session = self._classify_session(now)
        result.session = self._session.value

        # Check circuit breaker before proceeding
        if self._circuit_breaker.kill_switch_active:
            logger.warning("KILL SWITCH ACTIVE — skipping execution phases")
            # Still run monitoring + learning for awareness
            result.risk_level = RiskLevel.KILL_SWITCH.value
            self._run_phase_safe(LoopPhase.LEARNING, self.run_learning_phase, result)
            self._run_phase_safe(LoopPhase.MONITORING, self.run_monitoring_phase, result)
            result.total_duration_ms = (time.monotonic() - t_start) * 1000
            self._record_heartbeat(result)
            return result

        # Phase 1: DATA (every tick)
        self._run_phase_safe(LoopPhase.DATA, self.run_data_phase, result)

        # Phase 2: SIGNALS (1-min cadence)
        if self._should_run_cadence(self._last_signal_time, _SIGNAL_CADENCE):
            self._run_phase_safe(LoopPhase.SIGNALS, self.run_signal_phase, result)
            self._last_signal_time = now

        # Phase 3: INTELLIGENCE (5-min cadence)
        if self._should_run_cadence(self._last_intelligence_time, _INTELLIGENCE_CADENCE):
            self._run_phase_safe(LoopPhase.INTELLIGENCE, self.run_intelligence_phase, result)
            self._last_intelligence_time = now

        # Phase 4: DECISION (on signal change)
        if self._signals_changed():
            self._run_phase_safe(LoopPhase.DECISION, self.run_decision_phase, result)

        # Phase 5: EXECUTION (on approved trades)
        if self._approved_trades:
            self._run_phase_safe(LoopPhase.EXECUTION, self.run_execution_phase, result)

        # Phase 6: LEARNING (continuous)
        self._run_phase_safe(LoopPhase.LEARNING, self.run_learning_phase, result)

        # Phase 7: MONITORING (5-min cadence)
        if self._should_run_cadence(self._last_monitoring_time, _MONITORING_CADENCE):
            self._run_phase_safe(LoopPhase.MONITORING, self.run_monitoring_phase, result)
            self._last_monitoring_time = now

        # Finalize heartbeat
        result.total_duration_ms = (time.monotonic() - t_start) * 1000
        result.risk_level = self._circuit_breaker.level.value
        self._record_heartbeat(result)
        self._last_heartbeat_time = now

        logger.info(
            "--- Heartbeat #%d complete: %.1fms  signals=%d  trades=%d/%d  errors=%d ---",
            self._iteration, result.total_duration_ms,
            result.signals_generated, result.trades_approved,
            result.trades_executed, len(result.errors),
        )

        return result

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------
    def run_data_phase(self) -> PhaseResult:
        """Phase 1: Data ingestion and pooling.

        Runs DataIngestionOrchestrator for continuous data feed, then
        routes through UniversalDataPool for normalization and distribution.
        Falls back to UniverseEngine.load() if dedicated components absent.
        """
        pr = PhaseResult(phase=LoopPhase.DATA.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()

        items = 0

        # Primary: dedicated ingestion orchestrator
        ingestion = self._get("data_ingestion")
        if ingestion:
            try:
                ing_result = ingestion.run_continuous_loop()
                items += ing_result.get("records_ingested", 0) if isinstance(ing_result, dict) else 0
                pr.data["ingestion"] = "ok"
            except Exception as exc:
                pr.data["ingestion_error"] = str(exc)
                logger.warning("DataIngestion error: %s", exc)

        # Data pool routing
        pool = self._get("data_pool")
        if pool:
            try:
                pool_result = pool.pool_all()
                items += pool_result.get("records_pooled", 0) if isinstance(pool_result, dict) else 0
                pr.data["pool"] = "ok"
            except Exception as exc:
                pr.data["pool_error"] = str(exc)
                logger.warning("DataPool error: %s", exc)

        # Fallback: load universe directly
        universe = self._get("universe")
        if universe:
            try:
                universe.load()
                size = universe.size() if hasattr(universe, "size") else 0
                pr.data["universe_size"] = size
                items += size
            except Exception as exc:
                pr.data["universe_error"] = str(exc)
                logger.warning("Universe load error: %s", exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_signal_phase(self) -> PhaseResult:
        """Phase 2: Signal generation from all signal engines.

        Runs at 1-minute cadence:
            - FedLiquidityPlumbing.update()
            - MacroEngine.analyze()
            - MetadronCube.compute()
            - SecurityAnalysisEngine.analyze()
        """
        pr = PhaseResult(phase=LoopPhase.SIGNALS.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        signals_count = 0

        # Fed Liquidity Plumbing
        fed = self._get("fed_liquidity")
        if fed:
            try:
                fed_result = fed.update() if hasattr(fed, "update") else None
                pr.data["fed_liquidity"] = "updated"
                signals_count += 1
            except Exception as exc:
                pr.data["fed_liquidity_error"] = str(exc)
                logger.warning("FedLiquidity error: %s", exc)

        # Macro Engine
        macro = self._get("macro_engine")
        if macro:
            try:
                snap = macro.analyze()
                self._last_macro_snapshot = snap
                pr.data["macro_regime"] = snap.regime.value if hasattr(snap, "regime") else str(snap)
                pr.data["macro_vix"] = getattr(snap, "vix", 0.0)
                signals_count += 1
                logger.info("Macro regime: %s  VIX: %.1f", pr.data["macro_regime"], pr.data["macro_vix"])
            except Exception as exc:
                pr.data["macro_error"] = str(exc)
                logger.warning("MacroEngine error: %s", exc)

        # MetadronCube
        cube = self._get("metadron_cube")
        if cube and self._last_macro_snapshot:
            try:
                cube_out = cube.compute(self._last_macro_snapshot)
                self._last_cube_output = cube_out
                pr.data["cube_regime"] = cube_out.regime.value if hasattr(cube_out, "regime") else str(cube_out)
                pr.data["cube_target_beta"] = getattr(cube_out, "target_beta", 0.0)
                signals_count += 1
                logger.info(
                    "Cube regime: %s  target_beta: %.3f",
                    pr.data["cube_regime"], pr.data["cube_target_beta"],
                )
            except Exception as exc:
                pr.data["cube_error"] = str(exc)
                logger.warning("MetadronCube error: %s", exc)

        # Security Analysis Engine
        sa = self._get("security_analysis")
        if sa:
            try:
                universe = self._get("universe")
                tickers = []
                if universe and hasattr(universe, "get_all"):
                    tickers = [s.ticker for s in universe.get_all()[:50]]
                if tickers:
                    sa_macro = {}
                    if self._last_macro_snapshot:
                        snap = self._last_macro_snapshot
                        sa_macro = {
                            "treasury_10y": getattr(snap, "treasury_10y", 0.045),
                            "sp500_pe": getattr(snap, "sp500_pe", 22.0),
                            "cape": getattr(snap, "cape", 28.0),
                            "hy_spread": getattr(snap, "hy_spread", 4.0),
                            "ig_spread": getattr(snap, "ig_spread", 1.5),
                            "vix": getattr(snap, "vix", 20.0),
                            "gdp_growth": getattr(snap, "gdp_growth", 0.02),
                            "cpi": getattr(snap, "cpi", 0.03),
                            "fedfunds": getattr(snap, "fedfunds", 0.05),
                        }
                    sa_result = sa.analyze(tickers, sa_macro, {})
                    pr.data["security_analysis_count"] = getattr(sa_result, "tickers_analyzed", 0)
                    signals_count += 1
            except Exception as exc:
                pr.data["security_analysis_error"] = str(exc)
                logger.warning("SecurityAnalysis error: %s", exc)

        # Store signals for change detection
        new_signals = {
            "macro_regime": pr.data.get("macro_regime", ""),
            "cube_regime": pr.data.get("cube_regime", ""),
            "cube_target_beta": pr.data.get("cube_target_beta", 0.0),
            "vix": pr.data.get("macro_vix", 0.0),
        }
        self._last_signals = new_signals

        pr.items_processed = signals_count
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_intelligence_phase(self) -> PhaseResult:
        """Phase 3: ML intelligence and agent scoring.

        Runs at 5-minute cadence:
            - AlphaOptimizer.optimize()
            - ML vote ensemble scoring
            - Agent sector bot scoring
        """
        pr = PhaseResult(phase=LoopPhase.INTELLIGENCE.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        # Alpha Optimizer
        alpha = self._get("alpha_optimizer")
        if alpha:
            try:
                # Get universe tickers for optimization
                universe = self._get("universe")
                tickers = []
                if universe and hasattr(universe, "get_all"):
                    tickers = [s.ticker for s in universe.get_all()[:100]]

                if tickers and hasattr(alpha, "optimize"):
                    alpha_out = alpha.optimize(tickers)
                    self._last_alpha_output = alpha_out
                    pr.data["alpha_signals"] = len(getattr(alpha_out, "signals", []))
                    pr.data["alpha_expected_return"] = getattr(alpha_out, "expected_return", 0.0)
                    items += pr.data["alpha_signals"]
                elif tickers and hasattr(alpha, "run"):
                    alpha_out = alpha.run(tickers)
                    self._last_alpha_output = alpha_out
                    items += 1
            except Exception as exc:
                pr.data["alpha_error"] = str(exc)
                logger.warning("AlphaOptimizer error: %s", exc)

        # ML Vote Ensemble (accessed via execution engine)
        exec_engine = self._get("execution_engine")
        if exec_engine and hasattr(exec_engine, "ensemble"):
            try:
                ensemble = exec_engine.ensemble
                if hasattr(ensemble, "vote") and self._last_alpha_output:
                    signals = getattr(self._last_alpha_output, "signals", [])
                    for sig in signals[:20]:  # Top 20 signals
                        ticker = getattr(sig, "ticker", "")
                        if ticker:
                            try:
                                vote = ensemble.vote(ticker)
                                items += 1
                            except Exception:
                                pass
                    pr.data["ensemble_votes"] = items
            except Exception as exc:
                pr.data["ensemble_error"] = str(exc)
                logger.warning("MLEnsemble error: %s", exc)

        # Agent sector bots scoring
        bots = self._get("sector_bots")
        if bots:
            try:
                if hasattr(bots, "score_all"):
                    bot_scores = bots.score_all()
                    pr.data["agent_scores"] = len(bot_scores) if bot_scores else 0
                    items += pr.data["agent_scores"]
                elif hasattr(bots, "run_all"):
                    bot_result = bots.run_all()
                    items += 1
            except Exception as exc:
                pr.data["agent_error"] = str(exc)
                logger.warning("SectorBots error: %s", exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_decision_phase(self) -> PhaseResult:
        """Phase 4: Trade decision evaluation.

        Triggered on signal change:
            - DecisionMatrix.evaluate()
            - BetaCorridor.check()
            - Options opportunistic scan
            - Futures beta hedge check
        """
        pr = PhaseResult(phase=LoopPhase.DECISION.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        approved = 0

        self._pending_trades.clear()
        self._approved_trades.clear()

        # Decision Matrix evaluation
        dm = self._get("decision_matrix")
        if dm and self._last_alpha_output:
            try:
                signals = getattr(self._last_alpha_output, "signals", [])
                cube_out = self._last_cube_output
                macro_snap = self._last_macro_snapshot

                for sig in signals[:30]:
                    ticker = getattr(sig, "ticker", "")
                    if not ticker:
                        continue
                    try:
                        if hasattr(dm, "evaluate"):
                            decision = dm.evaluate(
                                signal=sig,
                                cube_output=cube_out,
                                macro_snapshot=macro_snap,
                            )
                        elif hasattr(dm, "score"):
                            decision = dm.score(sig)
                        else:
                            continue

                        is_approved = False
                        if isinstance(decision, dict):
                            is_approved = decision.get("approved", False)
                        elif hasattr(decision, "approved"):
                            is_approved = decision.approved
                        elif hasattr(decision, "composite_score"):
                            is_approved = decision.composite_score >= 0.55

                        if is_approved:
                            self._approved_trades.append({
                                "ticker": ticker,
                                "signal": sig,
                                "decision": decision,
                                "timestamp": datetime.now().isoformat(),
                            })
                            approved += 1
                    except Exception as exc:
                        logger.debug("Decision eval failed for %s: %s", ticker, exc)

                pr.data["candidates_evaluated"] = len(signals[:30])
                pr.data["trades_approved"] = approved
            except Exception as exc:
                pr.data["decision_error"] = str(exc)
                logger.warning("DecisionMatrix error: %s", exc)

        # Beta Corridor check
        beta = self._get("beta_corridor")
        if beta:
            try:
                if hasattr(beta, "check"):
                    beta_action = beta.check()
                elif hasattr(beta, "compute"):
                    beta_action = beta.compute()
                else:
                    beta_action = None

                if beta_action:
                    action_str = getattr(beta_action, "action", "HOLD")
                    pr.data["beta_action"] = action_str
                    if action_str != "HOLD":
                        self._approved_trades.append({
                            "ticker": getattr(beta_action, "instrument", "SPY"),
                            "signal": beta_action,
                            "decision": {"type": "beta_hedge", "action": action_str},
                            "timestamp": datetime.now().isoformat(),
                        })
                        approved += 1
            except Exception as exc:
                pr.data["beta_error"] = str(exc)
                logger.warning("BetaCorridor error: %s", exc)

        # Options opportunistic scan
        options = self._get("options_engine")
        if options:
            try:
                if hasattr(options, "scan_opportunities"):
                    opts = options.scan_opportunities(
                        cube_output=self._last_cube_output,
                        macro_snapshot=self._last_macro_snapshot,
                    )
                    pr.data["options_opportunities"] = len(opts) if opts else 0
                    for opt in (opts or []):
                        self._approved_trades.append({
                            "ticker": getattr(opt, "underlying", ""),
                            "signal": opt,
                            "decision": {"type": "options"},
                            "timestamp": datetime.now().isoformat(),
                        })
                        approved += 1
            except Exception as exc:
                pr.data["options_error"] = str(exc)
                logger.debug("OptionsEngine scan error: %s", exc)

        pr.data["total_approved"] = approved
        pr.items_processed = approved
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_execution_phase(self) -> PhaseResult:
        """Phase 5: Order execution.

        Triggered when approved trades exist:
            - ExecutionEngine.execute() for equity trades
            - OptionsEngine for options trades
            - Beta management via BetaCorridor
        """
        pr = PhaseResult(phase=LoopPhase.EXECUTION.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        executed = 0

        # Check circuit breaker — block execution if risk too high
        if self._circuit_breaker.level in (RiskLevel.CRITICAL, RiskLevel.KILL_SWITCH):
            pr.data["blocked_by_circuit_breaker"] = True
            pr.data["risk_level"] = self._circuit_breaker.level.value
            logger.warning(
                "Execution BLOCKED by circuit breaker: %s",
                self._circuit_breaker.level.value,
            )
            self._approved_trades.clear()
            pr.duration_ms = (time.monotonic() - t0) * 1000
            pr.success = True
            return pr

        exec_engine = self._get("execution_engine")

        # Determine current regime for L7 routing
        regime = "TRENDING"
        if self._last_macro_snapshot and hasattr(self._last_macro_snapshot, "regime"):
            regime = str(getattr(self._last_macro_snapshot, "regime", "TRENDING"))

        for trade in self._approved_trades:
            trade_type = trade.get("decision", {}).get("type", "equity") if isinstance(
                trade.get("decision"), dict
            ) else "equity"

            try:
                # L7 Unified Execution Surface — routes ALL products
                if exec_engine and hasattr(exec_engine, "l7") and exec_engine.l7 is not None:
                    ticker = trade.get("ticker", "")
                    signal = trade.get("signal")
                    if not ticker:
                        continue

                    # Determine side + quantity from signal
                    alpha_pred = getattr(signal, "alpha_pred", 0.0)
                    weight = getattr(signal, "weight", 0.0)
                    qty = max(1, int(abs(weight) * 100)) if weight != 0 else 1

                    if trade_type == "options":
                        side = "BUY" if alpha_pred >= 0 else "SELL"
                        exec_engine.l7_submit(
                            ticker=ticker, side=side, quantity=qty,
                            signal_type=getattr(signal, "signal_type", "HOLD"),
                            regime=regime, product_type="OPTION",
                        )
                    elif trade_type == "beta_hedge":
                        action = getattr(signal, "action", "HOLD")
                        instrument = getattr(signal, "instrument", "SPY")
                        hedge_qty = getattr(signal, "quantity", 0)
                        if action != "HOLD" and hedge_qty > 0:
                            exec_engine.l7_submit(
                                ticker=instrument, side=action, quantity=hedge_qty,
                                signal_type="MICRO_PRICE_BUY" if action == "BUY" else "MICRO_PRICE_SELL",
                                regime=regime, product_type="FUTURE" if instrument in ("ES", "NQ", "VX") else "EQUITY",
                            )
                    else:
                        side = "BUY" if (alpha_pred > 0 or weight > 0) else "SELL"
                        if alpha_pred == 0 and weight == 0:
                            continue
                        exec_engine.l7_submit(
                            ticker=ticker, side=side, quantity=qty,
                            signal_type=getattr(signal, "signal_type", "HOLD"),
                            regime=regime,
                        )
                    executed += 1

                # Fallback: direct broker execution (when L7 not available)
                elif exec_engine:
                    if trade_type == "options":
                        options = self._get("options_engine")
                        if options and hasattr(options, "execute"):
                            options.execute(trade["signal"])
                            executed += 1
                        elif options and hasattr(options, "evaluate_strategy"):
                            options.evaluate_strategy(trade["signal"])
                            executed += 1

                    elif trade_type == "beta_hedge":
                        if hasattr(exec_engine, "broker"):
                            signal = trade["signal"]
                            action = getattr(signal, "action", "HOLD")
                            qty = getattr(signal, "quantity", 0)
                            instrument = getattr(signal, "instrument", "SPY")
                            if action != "HOLD" and qty > 0:
                                if action == "BUY":
                                    exec_engine.broker.buy(instrument, qty)
                                elif action == "SELL":
                                    exec_engine.broker.sell(instrument, qty)
                                executed += 1

                    else:
                        ticker = trade.get("ticker", "")
                        signal = trade.get("signal")
                        if ticker and hasattr(exec_engine, "broker"):
                            alpha_pred = getattr(signal, "alpha_pred", 0.0)
                            weight = getattr(signal, "weight", 0.0)
                            if alpha_pred > 0 or weight > 0:
                                exec_engine.broker.buy(ticker, max(1, int(abs(weight) * 100)))
                            elif alpha_pred < 0:
                                exec_engine.broker.sell(ticker, max(1, int(abs(weight) * 100)))
                            executed += 1

            except Exception as exc:
                pr.errors.append(f"{trade.get('ticker', '?')}: {exc}")
                logger.warning("Execution failed for %s: %s", trade.get("ticker"), exc)

        # L7 heartbeat (every iteration)
        if exec_engine and hasattr(exec_engine, "l7_heartbeat"):
            try:
                exec_engine.l7_heartbeat(regime=regime)
            except Exception as exc:
                logger.debug("L7 heartbeat error: %s", exc)

        pr.data["trades_executed"] = executed
        pr.data["trades_attempted"] = len(self._approved_trades)
        pr.items_processed = executed

        # Clear executed trades
        self._approved_trades.clear()

        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_learning_phase(self) -> PhaseResult:
        """Phase 6: Learning and feedback loops.

        Continuous:
            - LearningLoop.record() for signal outcomes
            - GSDPlugin.update_gradients()
            - PaulPlugin.store_pattern()
            - Agent scorecard updates
        """
        pr = PhaseResult(phase=LoopPhase.LEARNING.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        # Learning Loop — record recent outcomes
        ll = self._get("learning_loop")
        if ll:
            try:
                # Record regime feedback if we have macro data
                if self._last_macro_snapshot and hasattr(ll, "record_regime_feedback"):
                    snap = self._last_macro_snapshot
                    regime_str = snap.regime.value if hasattr(snap.regime, "value") else str(snap.regime)
                    if RegimeFeedback:
                        feedback = RegimeFeedback(
                            predicted_regime=regime_str,
                            actual_regime=regime_str,
                            timestamp=datetime.now().isoformat(),
                        )
                        ll.record_regime_feedback(feedback)
                        items += 1

                # Get execution engine trade log for signal outcomes
                exec_engine = self._get("execution_engine")
                if exec_engine and hasattr(exec_engine, "_trade_log"):
                    recent_trades = exec_engine._trade_log[-10:]
                    for trade in recent_trades:
                        if SignalOutcome and hasattr(ll, "record_signal_outcome"):
                            try:
                                outcome = SignalOutcome(
                                    ticker=trade.get("ticker", ""),
                                    signal_engine=trade.get("engine", ""),
                                    signal_type=trade.get("signal_type", ""),
                                    realized_pnl=trade.get("pnl", 0.0),
                                    was_correct=trade.get("pnl", 0.0) > 0,
                                )
                                ll.record_signal_outcome(outcome)
                                items += 1
                            except Exception:
                                pass
            except Exception as exc:
                pr.data["learning_loop_error"] = str(exc)
                logger.warning("LearningLoop error: %s", exc)

        # GSD Plugin gradient updates
        gsd = self._get("gsd_plugin")
        if gsd:
            try:
                if hasattr(gsd, "update_gradients"):
                    gsd.update_gradients()
                    items += 1
                    pr.data["gsd_updated"] = True
            except Exception as exc:
                pr.data["gsd_error"] = str(exc)
                logger.debug("GSDPlugin error: %s", exc)

        # Paul Plugin pattern storage
        paul = self._get("paul_plugin")
        if paul:
            try:
                if hasattr(paul, "store_pattern"):
                    paul.store_pattern()
                    items += 1
                    pr.data["paul_updated"] = True
            except Exception as exc:
                pr.data["paul_error"] = str(exc)
                logger.debug("PaulPlugin error: %s", exc)

        # Agent scorecard updates
        scorecard = self._get("agent_scorecard")
        if scorecard:
            try:
                if hasattr(scorecard, "update"):
                    scorecard.update()
                    items += 1
                elif hasattr(scorecard, "refresh"):
                    scorecard.refresh()
                    items += 1
            except Exception as exc:
                pr.data["scorecard_error"] = str(exc)
                logger.debug("AgentScorecard error: %s", exc)

        # PaulOrchestrator — periodic enforcement + dynamic agent management
        paul_orch = self._get("paul_orchestrator")
        if paul_orch:
            try:
                # Run periodic enforcement check (herding, concentration, drift)
                periodic_result = paul_orch.run_periodic_check()
                pr.data["paul_orchestrator_check"] = True
                items += 1

                # Get ensemble adjustments for MLVoteEnsemble
                if self._last_macro_snapshot:
                    regime_str = ""
                    if hasattr(self._last_macro_snapshot, "regime"):
                        regime_str = (
                            self._last_macro_snapshot.regime.value
                            if hasattr(self._last_macro_snapshot.regime, "value")
                            else str(self._last_macro_snapshot.regime)
                        )
                    market_state = {"regime": regime_str}
                    adjustments = paul_orch.get_ensemble_adjustments(market_state)
                    pr.data["ensemble_adjustments"] = adjustments

                # Process recent trade outcomes through the orchestrator
                exec_engine = self._get("execution_engine")
                if exec_engine and hasattr(exec_engine, "_trade_log"):
                    recent = exec_engine._trade_log[-5:]
                    outcomes = []
                    for trade in recent:
                        outcomes.append({
                            "ticker": trade.get("ticker", ""),
                            "signal_engine": trade.get("engine", ""),
                            "realized_pnl": trade.get("pnl", 0.0),
                            "was_correct": trade.get("pnl", 0.0) > 0,
                            "direction": trade.get("side", ""),
                            "regime": trade.get("regime", ""),
                        })
                    if outcomes:
                        paul_orch.process_outcomes(outcomes)
                        items += len(outcomes)

            except Exception as exc:
                pr.data["paul_orchestrator_error"] = str(exc)
                logger.debug("PaulOrchestrator learning phase error: %s", exc)

        # GSD Workflow Bridge — feed metrics into structured state tracking
        gsd_wf = self._get("gsd_workflow")
        if gsd_wf:
            try:
                # Feed GSD gradient state
                gsd = self._get("gsd_plugin")
                if gsd and hasattr(gsd, "log_gradient_state"):
                    gsd_state = gsd.log_gradient_state()
                    gsd_wf.integrate_with_gradient_state(gsd_state)

                # Feed Paul pattern state
                paul = self._get("paul_plugin")
                if paul and hasattr(paul, "log_learning_state"):
                    paul_state = paul.log_learning_state()
                    gsd_wf.integrate_with_paul_state(paul_state)

                pr.data["gsd_workflow_updated"] = True
                items += 1
            except Exception as exc:
                pr.data["gsd_workflow_error"] = str(exc)
                logger.debug("GSD Workflow Bridge error: %s", exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_monitoring_phase(self) -> PhaseResult:
        """Phase 7: Monitoring and reporting.

        Runs at 5-minute cadence:
            - Portfolio P&L update
            - Risk check (circuit breaker evaluation)
            - Anomaly detection
            - Learning snapshot
        """
        pr = PhaseResult(phase=LoopPhase.MONITORING.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        # Portfolio P&L
        exec_engine = self._get("execution_engine")
        nav = self._initial_nav
        daily_pnl = 0.0
        if exec_engine and hasattr(exec_engine, "broker"):
            try:
                broker = exec_engine.broker
                if hasattr(broker, "nav"):
                    nav = broker.nav
                elif hasattr(broker, "get_nav"):
                    nav = broker.get_nav()
                elif hasattr(broker, "cash"):
                    nav = broker.cash

                daily_pnl = nav - self._initial_nav
                pr.data["nav"] = nav
                pr.data["daily_pnl"] = daily_pnl
                pr.data["daily_pnl_pct"] = daily_pnl / self._initial_nav if self._initial_nav > 0 else 0.0
                items += 1
            except Exception as exc:
                pr.data["nav_error"] = str(exc)

        # Risk check — update circuit breaker
        self._update_circuit_breaker(nav, daily_pnl)
        pr.data["risk_level"] = self._circuit_breaker.level.value
        pr.data["kill_switch"] = self._circuit_breaker.kill_switch_active

        # Anomaly detection
        anomaly = self._get("anomaly_detector")
        if anomaly:
            try:
                if hasattr(anomaly, "scan_returns"):
                    # Would need returns data; run a general scan if available
                    alerts = anomaly.get_alerts() if hasattr(anomaly, "get_alerts") else None
                    if alerts and hasattr(alerts, "get_critical"):
                        critical = alerts.get_critical()
                        pr.data["critical_anomalies"] = len(critical)
                        if critical:
                            logger.warning("ANOMALY DETECTOR: %d critical alerts", len(critical))
                        items += 1
            except Exception as exc:
                pr.data["anomaly_error"] = str(exc)
                logger.debug("AnomalyDetector error: %s", exc)

        # Learning snapshot
        ll = self._get("learning_loop")
        if ll and hasattr(ll, "get_snapshot"):
            try:
                snapshot = ll.get_snapshot()
                pr.data["learning_accuracy"] = getattr(snapshot, "overall_accuracy", 0.0)
                pr.data["learning_total_signals"] = getattr(snapshot, "total_signals", 0)
                items += 1
            except Exception as exc:
                pr.data["learning_snapshot_error"] = str(exc)

        # Portfolio analytics
        analytics = self._get("portfolio_analytics")
        if analytics:
            try:
                if hasattr(analytics, "compute"):
                    analytics.compute()
                    items += 1
            except Exception as exc:
                pr.data["analytics_error"] = str(exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    # ------------------------------------------------------------------
    # Daily schedule
    # ------------------------------------------------------------------
    def run_daily_schedule(self):
        """Execute time-based daily schedule actions.

        Called within the main loop to trigger session-specific routines:
            - Pre-market: full data refresh, overnight signals, SEC scan
            - Market open: full pipeline flush
            - Market close: EOD reconciliation, learning snapshot
            - After-hours: reduced frequency, earnings scan
        """
        now = datetime.now()
        session = self._classify_session(now)

        if session == MarketSession.PRE_MARKET:
            self._run_pre_market()
        elif session == MarketSession.MARKET_OPEN:
            self._run_market_open()
        elif session == MarketSession.MARKET_CLOSE:
            self._run_market_close()
        elif session == MarketSession.AFTER_HOURS:
            self._run_after_hours()

    def _run_pre_market(self):
        """Pre-market routine (08:00-09:30 ET).

        Full data refresh, overnight signal processing, SEC filing scan.
        """
        logger.info("PRE-MARKET ROUTINE — full data refresh")

        # Full universe reload
        universe = self._get("universe")
        if universe and hasattr(universe, "load"):
            try:
                universe.load()
                logger.info("Universe reloaded: %d securities",
                            universe.size() if hasattr(universe, "size") else 0)
            except Exception as exc:
                logger.warning("Pre-market universe reload failed: %s", exc)

        # Overnight macro signals
        macro = self._get("macro_engine")
        if macro:
            try:
                snap = macro.analyze()
                self._last_macro_snapshot = snap
                logger.info("Pre-market macro: regime=%s",
                            snap.regime.value if hasattr(snap, "regime") else "?")
            except Exception as exc:
                logger.warning("Pre-market macro failed: %s", exc)

        # Security analysis scan (SEC filings)
        sa = self._get("security_analysis")
        if sa and hasattr(sa, "scan_filings"):
            try:
                sa.scan_filings()
                logger.info("Pre-market SEC filing scan complete")
            except Exception as exc:
                logger.debug("Pre-market SEC scan failed: %s", exc)

    def _run_market_open(self):
        """Market open routine (09:30 ET).

        Full pipeline flush — runs every phase sequentially.
        """
        logger.info("=" * 60)
        logger.info("MARKET OPEN — full pipeline flush")
        logger.info("=" * 60)

        # Run the full execution engine pipeline if available
        exec_engine = self._get("execution_engine")
        if exec_engine and hasattr(exec_engine, "run_pipeline"):
            try:
                result = exec_engine.run_pipeline()
                logger.info("Market open pipeline complete: %d stages",
                            len(result.get("stages", {})))
            except Exception as exc:
                logger.error("Market open pipeline failed: %s", exc)

        # L7 market open — reset daily counters
        if exec_engine and hasattr(exec_engine, "l7_market_open"):
            try:
                exec_engine.l7_market_open()
                logger.info("L7 Unified Execution Surface: market open")
            except Exception as exc:
                logger.debug("L7 market open failed: %s", exc)

        # Force all cadence timers so everything runs on first intraday heartbeat
        self._last_signal_time = None
        self._last_intelligence_time = None
        self._last_monitoring_time = None

    def _run_market_close(self):
        """Market close routine (16:00 ET).

        EOD reconciliation, learning snapshot, next-day prep.
        """
        logger.info("=" * 60)
        logger.info("MARKET CLOSE — EOD reconciliation")
        logger.info("=" * 60)

        # Final learning snapshot
        ll = self._get("learning_loop")
        if ll and hasattr(ll, "get_snapshot"):
            try:
                snapshot = ll.get_snapshot()
                logger.info("EOD learning snapshot: accuracy=%.2f%% total_signals=%d",
                            getattr(snapshot, "overall_accuracy", 0.0) * 100,
                            getattr(snapshot, "total_signals", 0))
            except Exception as exc:
                logger.warning("EOD learning snapshot failed: %s", exc)

        # L7 market close — daily learning + pattern persistence
        exec_engine = self._get("execution_engine")
        if exec_engine and hasattr(exec_engine, "l7_market_close"):
            try:
                exec_engine.l7_market_close()
                logger.info("L7 Unified Execution Surface: market close")
            except Exception as exc:
                logger.debug("L7 market close failed: %s", exc)

        # Persist state
        if self._enable_persistence:
            self._persist_state()

        # Agent scorecard weekly update
        scorecard = self._get("agent_scorecard")
        if scorecard and hasattr(scorecard, "weekly_update"):
            try:
                scorecard.weekly_update()
            except Exception as exc:
                logger.debug("EOD scorecard update failed: %s", exc)

    def _run_after_hours(self):
        """After-hours routine (16:00-20:00 ET).

        Reduced frequency, earnings scan.
        """
        # Earnings scan via security analysis
        sa = self._get("security_analysis")
        if sa and hasattr(sa, "scan_earnings"):
            try:
                sa.scan_earnings()
            except Exception as exc:
                logger.debug("After-hours earnings scan failed: %s", exc)

    # ------------------------------------------------------------------
    # Overnight backtesting
    # ------------------------------------------------------------------
    def run_backtest_overnight(self) -> PhaseResult:
        """Run overnight backtesting and ML training.

        Executes:
            - QSTrader backtest runner on current strategy
            - ML model retraining
            - Pattern evolution
        """
        pr = PhaseResult(phase="BACKTEST_OVERNIGHT", timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        logger.info("OVERNIGHT — starting backtesting and ML training")

        # QSTrader backtest
        runner = self._get("backtest_runner")
        if runner:
            try:
                if hasattr(runner, "run"):
                    bt_result = runner.run()
                    pr.data["backtest"] = bt_result if isinstance(bt_result, dict) else {"status": "complete"}
                    items += 1
                elif hasattr(runner, "run_backtest"):
                    bt_result = runner.run_backtest()
                    pr.data["backtest"] = bt_result if isinstance(bt_result, dict) else {"status": "complete"}
                    items += 1
                logger.info("Overnight backtest complete")
            except Exception as exc:
                pr.data["backtest_error"] = str(exc)
                logger.warning("Overnight backtest failed: %s", exc)

        # ML model retraining via alpha optimizer
        alpha = self._get("alpha_optimizer")
        if alpha and hasattr(alpha, "retrain"):
            try:
                alpha.retrain()
                pr.data["ml_retrain"] = "complete"
                items += 1
            except Exception as exc:
                pr.data["ml_retrain_error"] = str(exc)
                logger.warning("Overnight ML retrain failed: %s", exc)

        # Pattern evolution via learning loop
        ll = self._get("learning_loop")
        if ll and hasattr(ll, "evolve_patterns"):
            try:
                ll.evolve_patterns()
                pr.data["pattern_evolution"] = "complete"
                items += 1
            except Exception as exc:
                pr.data["pattern_evolution_error"] = str(exc)

        # GSD/Paul plugin overnight learning
        gsd = self._get("gsd_plugin")
        if gsd and hasattr(gsd, "overnight_learn"):
            try:
                gsd.overnight_learn()
                items += 1
            except Exception as exc:
                logger.debug("GSD overnight learn failed: %s", exc)

        paul = self._get("paul_plugin")
        if paul and hasattr(paul, "overnight_learn"):
            try:
                paul.overnight_learn()
                items += 1
            except Exception as exc:
                logger.debug("Paul overnight learn failed: %s", exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True

        logger.info("Overnight session complete: %d tasks  %.1fms", items, pr.duration_ms)
        return pr

    # ------------------------------------------------------------------
    # Status and performance
    # ------------------------------------------------------------------
    def get_status(self) -> LoopStatus:
        """Return current loop status snapshot."""
        with self._lock:
            now = datetime.now()
            uptime = (now - self._started_at).total_seconds() if self._started_at else 0.0

            # Estimate time to next heartbeat
            next_hb = 0.0
            if self._last_heartbeat_time and self._state == LoopState.RUNNING:
                elapsed = (now - self._last_heartbeat_time).total_seconds()
                interval = self._get_current_interval()
                next_hb = max(0.0, interval - elapsed)

            # NAV from execution engine
            nav = self._initial_nav
            daily_pnl = 0.0
            exec_engine = self._get("execution_engine")
            if exec_engine and hasattr(exec_engine, "broker"):
                try:
                    broker = exec_engine.broker
                    if hasattr(broker, "nav"):
                        nav = broker.nav
                    elif hasattr(broker, "get_nav"):
                        nav = broker.get_nav()
                    daily_pnl = nav - self._initial_nav
                except Exception:
                    pass

            return LoopStatus(
                state=self._state.value,
                session=self._session.value,
                risk_level=self._circuit_breaker.level.value,
                current_phase=self._current_phase,
                iteration=self._iteration,
                uptime_seconds=uptime,
                last_heartbeat=self._last_heartbeat_time.isoformat() if self._last_heartbeat_time else "",
                next_heartbeat_in_seconds=next_hb,
                components_loaded=dict(self._component_status),
                kill_switch_active=self._circuit_breaker.kill_switch_active,
                performance=self._performance,
                last_macro_regime=(
                    self._last_macro_snapshot.regime.value
                    if self._last_macro_snapshot and hasattr(self._last_macro_snapshot, "regime")
                    else ""
                ),
                last_cube_regime=(
                    self._last_cube_output.regime.value
                    if self._last_cube_output and hasattr(self._last_cube_output, "regime")
                    else ""
                ),
                portfolio_nav=nav,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl / self._initial_nav if self._initial_nav > 0 else 0.0,
            )

    def get_performance(self) -> LoopPerformance:
        """Return cumulative loop performance metrics."""
        with self._lock:
            return self._performance

    # ------------------------------------------------------------------
    # Internal — main loop
    # ------------------------------------------------------------------
    def _run_loop(self):
        """Main loop body — runs on the background thread."""
        with self._lock:
            self._state = LoopState.RUNNING

        logger.info("LiveLoop thread started")

        # Track whether we've triggered session-specific routines
        _triggered_sessions: dict[str, bool] = {}

        while not self._stop_event.is_set():
            try:
                # Check pause
                if self._state == LoopState.PAUSED:
                    self._stop_event.wait(timeout=1.0)
                    continue

                now = datetime.now()
                session = self._classify_session(now)
                self._session = session

                # Trigger once-per-session routines
                session_key = f"{now.date()}_{session.value}"
                if session_key not in _triggered_sessions:
                    _triggered_sessions[session_key] = True
                    self.run_daily_schedule()

                    # Overnight backtesting
                    if session == MarketSession.OVERNIGHT:
                        self.run_backtest_overnight()

                # Run heartbeat
                try:
                    result = self.heartbeat()

                    # Reset consecutive errors on success
                    if not result.errors:
                        self._performance.consecutive_errors = 0

                except Exception as exc:
                    self._performance.total_errors += 1
                    self._performance.consecutive_errors += 1
                    self._performance.max_consecutive_errors = max(
                        self._performance.max_consecutive_errors,
                        self._performance.consecutive_errors,
                    )
                    logger.error("Heartbeat exception: %s\n%s", exc, traceback.format_exc())

                    # Auto-restart guard
                    if self._performance.consecutive_errors >= self._max_consecutive_errors:
                        logger.critical(
                            "FATAL: %d consecutive errors — entering ERROR state",
                            self._performance.consecutive_errors,
                        )
                        with self._lock:
                            self._state = LoopState.ERROR
                        break

                # Periodic state persistence
                if self._enable_persistence and self._should_run_cadence(
                    self._last_state_persist_time, _STATE_PERSISTENCE_INTERVAL
                ):
                    self._persist_state()
                    self._last_state_persist_time = now

                # Sleep until next heartbeat
                interval = self._get_current_interval()
                self._stop_event.wait(timeout=interval)

            except Exception as exc:
                logger.error("Loop iteration fatal error: %s\n%s", exc, traceback.format_exc())
                self._performance.total_errors += 1
                self._performance.consecutive_errors += 1
                time.sleep(_RESTART_COOLDOWN_SECONDS)

        logger.info("LiveLoop thread exiting")

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------
    def _classify_session(self, now: datetime) -> MarketSession:
        """Determine current market session from wall clock time."""
        # Weekend check (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            return MarketSession.WEEKEND

        t = now.time()

        if t < _PRE_MARKET_START:
            return MarketSession.OVERNIGHT
        elif t < _MARKET_OPEN:
            return MarketSession.PRE_MARKET
        elif t < dt_time(9, 31):
            return MarketSession.MARKET_OPEN
        elif t < _MARKET_CLOSE:
            return MarketSession.INTRADAY
        elif t < dt_time(16, 1):
            return MarketSession.MARKET_CLOSE
        elif t < _AFTER_HOURS_END:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.OVERNIGHT

    def _get_current_interval(self) -> float:
        """Return heartbeat interval based on current session."""
        if self._session == MarketSession.INTRADAY:
            return self._heartbeat_interval
        elif self._session in (MarketSession.PRE_MARKET, MarketSession.MARKET_OPEN):
            return self._heartbeat_interval
        elif self._session == MarketSession.AFTER_HOURS:
            return _AFTER_HOURS_INTERVAL
        elif self._session == MarketSession.OVERNIGHT:
            return _OVERNIGHT_INTERVAL
        elif self._session == MarketSession.WEEKEND:
            return _OVERNIGHT_INTERVAL
        else:
            return self._heartbeat_interval

    def _should_run_cadence(self, last_time: Optional[datetime], cadence_seconds: float) -> bool:
        """Check whether enough time has elapsed since last run."""
        if last_time is None:
            return True
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= cadence_seconds

    def _signals_changed(self) -> bool:
        """Detect whether signals have materially changed since last decision.

        Returns True if regime changed, beta target shifted, or VIX moved
        beyond a threshold.
        """
        if not self._last_signals:
            return False

        prev = getattr(self, "_prev_signals", {})
        if not prev:
            self._prev_signals = dict(self._last_signals)
            return True  # First time — trigger decision

        changed = False

        # Regime change
        if self._last_signals.get("macro_regime") != prev.get("macro_regime"):
            changed = True
        if self._last_signals.get("cube_regime") != prev.get("cube_regime"):
            changed = True

        # Beta target shift > 0.05
        beta_delta = abs(
            self._last_signals.get("cube_target_beta", 0.0) -
            prev.get("cube_target_beta", 0.0)
        )
        if beta_delta > 0.05:
            changed = True

        # VIX move > 2 points
        vix_delta = abs(
            self._last_signals.get("vix", 0.0) - prev.get("vix", 0.0)
        )
        if vix_delta > 2.0:
            changed = True

        # Always run decision at least every 5 minutes
        if hasattr(self, "_last_decision_time") and self._last_decision_time:
            if (datetime.now() - self._last_decision_time).total_seconds() > 300:
                changed = True
        else:
            changed = True

        if changed:
            self._prev_signals = dict(self._last_signals)
            self._last_decision_time = datetime.now()

        return changed

    def _run_phase_safe(
        self,
        phase: LoopPhase,
        phase_fn: Callable[[], PhaseResult],
        heartbeat_result: HeartbeatResult,
    ):
        """Execute a phase with error handling, timing, and recording."""
        self._current_phase = phase.value
        try:
            pr = phase_fn()
            heartbeat_result.phases[phase.value] = pr

            # Accumulate into heartbeat totals
            if phase == LoopPhase.SIGNALS:
                heartbeat_result.signals_generated += pr.items_processed
            elif phase == LoopPhase.DECISION:
                heartbeat_result.trades_approved += pr.items_processed
            elif phase == LoopPhase.EXECUTION:
                heartbeat_result.trades_executed += pr.items_processed

            # Track phase performance
            phase_name = phase.value
            self._performance.phase_total_ms[phase_name] = (
                self._performance.phase_total_ms.get(phase_name, 0.0) + pr.duration_ms
            )
            self._performance.phase_call_count[phase_name] = (
                self._performance.phase_call_count.get(phase_name, 0) + 1
            )
            count = self._performance.phase_call_count[phase_name]
            total = self._performance.phase_total_ms[phase_name]
            self._performance.phase_avg_ms[phase_name] = total / count if count > 0 else 0.0

            if not pr.success:
                heartbeat_result.errors.append(f"{phase.value}: {pr.error}")

        except Exception as exc:
            error_msg = f"{phase.value}: {exc}"
            heartbeat_result.errors.append(error_msg)
            logger.error("Phase %s failed: %s", phase.value, exc)

            heartbeat_result.phases[phase.value] = PhaseResult(
                phase=phase.value,
                success=False,
                error=str(exc),
                timestamp=datetime.now().isoformat(),
            )

        self._current_phase = ""

    def _record_heartbeat(self, result: HeartbeatResult):
        """Record heartbeat result into performance tracking."""
        with self._lock:
            self._heartbeat_history.append(result)
            perf = self._performance

            perf.total_iterations = self._iteration
            perf.last_heartbeat = result.timestamp
            perf.total_signals += result.signals_generated
            perf.total_trades_approved += result.trades_approved
            perf.total_trades_executed += result.trades_executed
            perf.total_errors += len(result.errors)

            # Duration tracking
            ms = result.total_duration_ms
            perf.max_heartbeat_ms = max(perf.max_heartbeat_ms, ms)
            perf.min_heartbeat_ms = min(perf.min_heartbeat_ms, ms)

            if self._started_at:
                perf.total_runtime_seconds = (datetime.now() - self._started_at).total_seconds()

            if perf.total_iterations > 0:
                perf.avg_heartbeat_ms = (
                    (perf.avg_heartbeat_ms * (perf.total_iterations - 1) + ms)
                    / perf.total_iterations
                )

            # Uptime calculation
            if perf.total_runtime_seconds > 0:
                expected_iterations = perf.total_runtime_seconds / self._heartbeat_interval
                if expected_iterations > 0:
                    perf.uptime_pct = min(100.0, (perf.total_iterations / expected_iterations) * 100)

    def _update_circuit_breaker(self, nav: float, daily_pnl: float):
        """Evaluate risk circuit breaker state."""
        cb = self._circuit_breaker
        cb.last_check = datetime.now().isoformat()
        cb.triggers.clear()

        # Drawdown check
        if self._initial_nav > 0:
            cb.current_drawdown_pct = abs(min(0, daily_pnl)) / self._initial_nav
        else:
            cb.current_drawdown_pct = 0.0

        cb.max_drawdown_pct = max(cb.max_drawdown_pct, cb.current_drawdown_pct)

        # VIX from last macro snapshot
        if self._last_macro_snapshot:
            cb.vix_level = getattr(self._last_macro_snapshot, "vix", 20.0)

        # Evaluate risk level
        level = RiskLevel.NORMAL

        if cb.current_drawdown_pct >= cb.DRAWDOWN_KILL:
            level = RiskLevel.KILL_SWITCH
            cb.triggers.append(f"DRAWDOWN {cb.current_drawdown_pct:.1%} >= {cb.DRAWDOWN_KILL:.1%}")
        elif cb.current_drawdown_pct >= cb.DRAWDOWN_CRITICAL:
            level = RiskLevel.CRITICAL
            cb.triggers.append(f"DRAWDOWN {cb.current_drawdown_pct:.1%}")
        elif cb.current_drawdown_pct >= cb.DRAWDOWN_HIGH:
            level = max(level, RiskLevel.HIGH)
            cb.triggers.append(f"DRAWDOWN {cb.current_drawdown_pct:.1%}")
        elif cb.current_drawdown_pct >= cb.DRAWDOWN_ELEVATED:
            level = max(level, RiskLevel.ELEVATED)

        # VIX check
        if cb.vix_level >= cb.VIX_CRITICAL:
            level = max(level, RiskLevel.CRITICAL)
            cb.triggers.append(f"VIX {cb.vix_level:.1f}")
        elif cb.vix_level >= cb.VIX_HIGH:
            level = max(level, RiskLevel.HIGH)
            cb.triggers.append(f"VIX {cb.vix_level:.1f}")
        elif cb.vix_level >= cb.VIX_ELEVATED:
            level = max(level, RiskLevel.ELEVATED)

        # MetadronCube Kill-Switch: HY OAS +35bp & VIX term flat/inverted & breadth < 50%
        if (cb.hy_spread_change_bps >= cb.HY_SPREAD_KILL_BPS and
                cb.breadth_pct < cb.BREADTH_KILL_PCT and
                cb.vix_level >= cb.VIX_HIGH):
            level = RiskLevel.KILL_SWITCH
            cb.triggers.append("CUBE_KILL_SWITCH: HY+VIX+Breadth")

        cb.level = level
        cb.kill_switch_active = (level == RiskLevel.KILL_SWITCH)

        if cb.kill_switch_active:
            logger.critical("KILL SWITCH ACTIVATED: %s", ", ".join(cb.triggers))
        elif level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            logger.warning("Risk level %s: %s", level.value, ", ".join(cb.triggers))

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _persist_state(self):
        """Save loop state and performance to disk for restart recovery."""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)

            state = {
                "iteration": self._iteration,
                "session": self._session.value,
                "risk_level": self._circuit_breaker.level.value,
                "kill_switch": self._circuit_breaker.kill_switch_active,
                "started_at": self._started_at.isoformat() if self._started_at else "",
                "persisted_at": datetime.now().isoformat(),
                "last_signals": self._last_signals,
                "circuit_breaker": {
                    "max_drawdown_pct": self._circuit_breaker.max_drawdown_pct,
                    "consecutive_losses": self._circuit_breaker.consecutive_losses,
                },
            }

            _STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

            perf = asdict(self._performance)
            _PERFORMANCE_FILE.write_text(json.dumps(perf, indent=2, default=str))

            logger.debug("State persisted: iteration=%d", self._iteration)

        except Exception as exc:
            logger.warning("State persistence failed: %s", exc)

    def _restore_state(self):
        """Restore loop state from disk after restart."""
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text())
                self._iteration = data.get("iteration", 0)
                self._last_signals = data.get("last_signals", {})
                self._performance.restarts += 1

                cb_data = data.get("circuit_breaker", {})
                self._circuit_breaker.max_drawdown_pct = cb_data.get("max_drawdown_pct", 0.0)
                self._circuit_breaker.consecutive_losses = cb_data.get("consecutive_losses", 0)

                logger.info(
                    "State restored: iteration=%d restarts=%d",
                    self._iteration, self._performance.restarts,
                )

            if _PERFORMANCE_FILE.exists():
                perf_data = json.loads(_PERFORMANCE_FILE.read_text())
                # Restore cumulative counters
                self._performance.total_signals = perf_data.get("total_signals", 0)
                self._performance.total_trades_approved = perf_data.get("total_trades_approved", 0)
                self._performance.total_trades_executed = perf_data.get("total_trades_executed", 0)
                self._performance.total_errors = perf_data.get("total_errors", 0)

        except Exception as exc:
            logger.warning("State restore failed (starting fresh): %s", exc)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        loaded = sum(self._component_status.values())
        total = len(self._component_status)
        return (
            f"LiveLoopOrchestrator("
            f"state={self._state.value}, "
            f"session={self._session.value}, "
            f"iteration={self._iteration}, "
            f"components={loaded}/{total})"
        )
