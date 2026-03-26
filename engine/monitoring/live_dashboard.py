"""
Live terminal dashboard for Metadron Capital trading engine.

Rich-based real-time display with scanner, positions, signals,
sector heatmap, portfolio summary, risk metrics, and performance.
Falls back to pure ASCII if Rich is not installed.
"""

import logging
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing Rich; fall back gracefully
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.info("Rich library not available — using plain-text dashboard fallback.")

# ---------------------------------------------------------------------------
# 550+ symbol scanner universe  (S&P 500 core + key mid-caps)
# ---------------------------------------------------------------------------
SCANNER_UNIVERSE: List[str] = [
    # -- Technology --
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO",
    "ACN", "INTC", "IBM", "INTU", "TXN", "QCOM", "AMAT", "NOW", "PANW",
    "ADI", "LRCX", "MU", "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "ROP",
    "NXPI", "MPWR", "ON", "KEYS", "EPAM", "SEDG", "ENPH", "TER", "ZBRA",
    "GLOB", "SMCI", "CRWD", "DDOG", "ZS", "NET", "MDB", "SNOW", "PLTR",
    "TEAM", "HUBS", "WDAY", "VEEV", "ANSS", "FIVN", "OKTA", "TTD", "BILL",
    "ESTC", "CFLT", "PATH", "S", "DOCN", "GTLB", "MNDY", "PCOR", "IOT",
    # -- Financials --
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    "SPGI", "C", "CB", "MMC", "PGR", "ICE", "CME", "AON", "MCO", "MET",
    "AIG", "AFL", "TRV", "ALL", "PRU", "BK", "USB", "PNC", "TFC", "STT",
    "FITB", "HBAN", "MTB", "CFG", "RF", "KEY", "NTRS", "SIVB", "ZION",
    "CMA", "FRC", "ALLY", "COIN", "HOOD", "SOFI", "SQ", "PYPL", "FIS",
    "FISV", "GPN", "WEX", "DFS", "COF", "SYF", "NDAQ",
    # -- Healthcare --
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "MDT", "ELV", "ISRG", "GILD", "VRTX", "REGN", "SYK",
    "BSX", "ZTS", "BDX", "CI", "HCA", "MCK", "DXCM", "IDXX", "IQV",
    "A", "MTD", "RMD", "EW", "ALGN", "HOLX", "WAT", "WST", "CRL",
    "PODD", "TFX", "INCY", "MRNA", "BNTX", "EXAS", "HZNP", "SGEN",
    "PCVX", "CRSP", "BEAM", "NTLA", "EDIT", "IONS", "ALNY", "RARE",
    # -- Consumer Discretionary --
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
    "CMG", "ORLY", "AZO", "ROST", "DHI", "LEN", "PHM", "GM", "F",
    "MAR", "HLT", "ABNB", "UBER", "LYFT", "DASH", "DKNG", "WYNN",
    "MGM", "LVS", "RCL", "CCL", "NCLH", "EXPE", "ETSY", "W", "EBAY",
    "CPRT", "POOL", "TSCO", "BBY", "DG", "DLTR", "FIVE", "ULTA", "LULU",
    "TPR", "RL", "PVH", "HAS", "GPS", "ANF", "DECK", "CROX", "BIRD",
    # -- Communication Services --
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "ATVI", "TTWO", "RBLX", "MTCH", "SNAP", "PINS", "ZM", "SPOT",
    "WBD", "PARA", "FOX", "FOXA", "LYV", "IMAX", "ROKU", "BILI",
    # -- Industrials --
    "CAT", "UNP", "HON", "UPS", "RTX", "BA", "DE", "LMT", "GE",
    "ADP", "MMM", "GD", "NOC", "CSX", "NSC", "FDX", "WM", "ETN",
    "ITW", "EMR", "PH", "ROK", "CMI", "PCAR", "TT", "CARR", "OTIS",
    "IR", "FAST", "CTAS", "PAYX", "VRSK", "CPRT", "ODFL", "JBHT",
    "XPO", "CHRW", "DAL", "UAL", "LUV", "AAL", "GNRC", "PWR", "BLDR",
    "URI", "WAB", "SWK", "HWM", "AXON", "TDG",
    # -- Consumer Staples --
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL",
    "EL", "KMB", "GIS", "K", "HSY", "SJM", "CAG", "CPB", "HRL",
    "MKC", "TSN", "KHC", "KR", "SYY", "ADM", "BG", "STZ", "TAP",
    "SAM", "MNST", "COKE", "CELH", "USFD",
    # -- Energy --
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD",
    "OXY", "HES", "DVN", "FANG", "HAL", "BKR", "CTRA", "MRO", "APA",
    "EQT", "AR", "RRC", "OVV", "TRGP", "WMB", "KMI", "OKE", "ET",
    "EPD", "MPLX", "PAA", "LNG", "DINO", "PBF", "VNOM", "MTDR",
    # -- Utilities --
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC",
    "ED", "ES", "AWK", "AEE", "DTE", "PPL", "FE", "CMS", "EIX",
    "EVRG", "AES", "PNW", "NI", "LNT", "ATO", "OGE", "NRG",
    # -- Real Estate --
    "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "DLR", "WELL",
    "AVB", "EQR", "VTR", "ARE", "MAA", "ESS", "UDR", "CPT", "PEAK",
    "HST", "INVH", "CUBE", "EXR", "REXR", "GLPI", "VICI", "SBAC",
    # -- Materials --
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD",
    "PPG", "VMC", "MLM", "ALB", "CF", "MOS", "IFF", "CE", "EMN",
    "BALL", "PKG", "IP", "WRK", "SEE", "AVY", "FMC", "CTVA",
    # -- Additional Mid-Caps & High-Beta --
    "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "GOEV", "ARVL",
    "CHPT", "BLNK", "EVGO", "QS", "LAZR", "VLDR", "MVIS", "LIDR",
    "AFRM", "UPST", "LMND", "ROOT", "OPEN", "RDFN", "CVNA", "CARG",
    "CELH", "MNST", "FRPT", "RUN", "NOVA", "ARRY", "STEM", "BEEM",
    "SPWR", "FSLR", "JKS", "DQ", "MAXN", "FLNC", "ASTS", "IRDM",
    "RKLB", "ASTR", "LUNR", "RDW", "VSAT", "GSAT",
    "IONQ", "RGTI", "QUBT", "ACHR", "JOBY", "LILM", "BLDE", "EVTL",
    "GRAB", "SE", "SHOP", "MELI", "NU", "BABA", "JD", "PDD", "BIDU",
    "TME", "FUTU", "TIGR", "CPNG",
    "AI", "BBAI", "SOUN", "PRCT", "ISRG", "NARI", "RXRX", "TWST",
    "DNLI", "FATE", "LEGN", "IMVT", "RCKT", "BLUE", "SRPT", "BMRN",
    "ARGX", "HALO", "FOLD", "DVAX", "IOVA", "RVMD", "KRTX", "CRNX",
    "APP", "DUOL", "TOST", "BROS", "CAVA", "CART", "INST", "BRZE",
    "AUR", "TER", "MARA", "RIOT", "HUT", "BTBT", "CLSK", "CIFR",
    "IREN", "WULF", "BITF", "CORZ",
]

# Deduplicate while preserving order
_seen = set()
_unique: List[str] = []
for _sym in SCANNER_UNIVERSE:
    if _sym not in _seen:
        _seen.add(_sym)
        _unique.append(_sym)
SCANNER_UNIVERSE = _unique
del _seen, _unique, _sym

# GICS sectors for heatmap
GICS_SECTORS: List[str] = [
    "Technology", "Financials", "Healthcare", "Consumer Discretionary",
    "Communication Services", "Industrials", "Consumer Staples",
    "Energy", "Utilities", "Real Estate", "Materials",
]


# ──────────────────────────────────────────────────────────────────────
# Enum: dashboard panel identifiers
# ──────────────────────────────────────────────────────────────────────
class DashboardPanel(Enum):
    SCANNER = "scanner"
    POSITIONS = "positions"
    SIGNALS = "signals"
    HEATMAP = "heatmap"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    PERFORMANCE = "performance"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _sparkline(values: List[float], width: int = 30) -> str:
    """Return a tiny ASCII sparkline for *values*."""
    if not values:
        return ""
    blocks = " _.-~*^"
    lo, hi = min(values), max(values)
    spread = hi - lo if hi != lo else 1.0
    scaled = [int((v - lo) / spread * (len(blocks) - 1)) for v in values]
    # Downsample / upsample to *width*
    if len(scaled) > width:
        step = len(scaled) / width
        scaled = [scaled[int(i * step)] for i in range(width)]
    return "".join(blocks[s] for s in scaled)


def _pnl_color(value: float) -> str:
    """Return ANSI escape for green (positive) or red (negative)."""
    if value > 0:
        return "\033[92m"
    elif value < 0:
        return "\033[91m"
    return "\033[0m"


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"


def _pad(text: str, width: int) -> str:
    """Left-pad *text* to *width*, ignoring ANSI codes for length."""
    import re
    visible = len(re.sub(r"\033\[[0-9;]*m", "", text))
    return text + " " * max(0, width - visible)


def _rpad(text: str, width: int) -> str:
    """Right-pad *text* to *width*."""
    import re
    visible = len(re.sub(r"\033\[[0-9;]*m", "", text))
    return " " * max(0, width - visible) + text


# ──────────────────────────────────────────────────────────────────────
# LiveDashboard
# ──────────────────────────────────────────────────────────────────────
class LiveDashboard:
    """Rich-based live terminal dashboard with ASCII fallback."""

    def __init__(self, refresh_interval: float = 5.0) -> None:
        self.refresh_interval = refresh_interval
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._live: Optional[Any] = None  # Rich Live instance when available
        logger.info(
            "LiveDashboard initialised  (rich=%s, refresh=%.1fs, universe=%d symbols)",
            RICH_AVAILABLE,
            refresh_interval,
            len(SCANNER_UNIVERSE),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the dashboard refresh loop in a background thread."""
        if self._running:
            logger.warning("Dashboard already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="dashboard")
        self._thread.start()
        logger.info("Dashboard started.")

    def stop(self) -> None:
        """Signal the dashboard thread to stop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.refresh_interval + 2)
            self._thread = None
        logger.info("Dashboard stopped.")

    def update_data(self, data: dict) -> None:
        """Push fresh pipeline data to the dashboard."""
        with self._lock:
            self._data.update(data)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        if RICH_AVAILABLE:
            self._loop_rich()
        else:
            self._loop_plain()

    def _loop_rich(self) -> None:
        """Use Rich Live display for auto-refreshing panels."""
        console = Console()
        try:
            with Live(console=console, refresh_per_second=1.0 / self.refresh_interval) as live:
                self._live = live
                while self._running:
                    try:
                        with self._lock:
                            snapshot = dict(self._data)
                        renderable = self._build_rich_layout(snapshot)
                        live.update(renderable)
                    except Exception:
                        logger.exception("Error rendering Rich dashboard")
                    time.sleep(self.refresh_interval)
        except Exception:
            logger.exception("Rich live loop failed — falling back to plain text")
            self._loop_plain()
        finally:
            self._live = None

    def _loop_plain(self) -> None:
        """Fallback: clear terminal and print ASCII dashboard."""
        while self._running:
            try:
                with self._lock:
                    snapshot = dict(self._data)
                output = self.format_full_dashboard(snapshot)
                # Clear screen and print
                print("\033[2J\033[H" + output, flush=True)
            except Exception:
                logger.exception("Error rendering plain dashboard")
            time.sleep(self.refresh_interval)

    # ------------------------------------------------------------------
    # Rich layout builder
    # ------------------------------------------------------------------
    def _build_rich_layout(self, data: dict) -> "Layout":
        """Compose Rich Layout with all panels."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="upper", ratio=2),
            Layout(name="signals", size=14),
            Layout(name="heatmap", size=10),
            Layout(name="footer", ratio=1),
        )
        layout["upper"].split_row(
            Layout(name="scanner", ratio=1),
            Layout(name="positions", ratio=1),
        )
        layout["footer"].split_row(
            Layout(name="portfolio", ratio=1),
            Layout(name="risk", ratio=1),
            Layout(name="performance", ratio=1),
        )

        # Header
        nav = data.get("portfolio", {}).get("nav", 0)
        regime = data.get("regime", "UNKNOWN")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text(
            f"  METADRON CAPITAL  |  {ts}  |  NAV ${nav:,.0f}  |  Regime: {regime}  |  Universe: {len(SCANNER_UNIVERSE)} symbols",
            style="bold cyan",
        )
        layout["header"].update(Panel(header_text, style="bold blue"))

        # Panels
        layout["scanner"].update(
            Panel(self._render_scanner_panel_rich(data.get("scanner", {})), title="Market Scanner")
        )
        layout["positions"].update(
            Panel(self._render_positions_panel_rich(data.get("positions", [])), title="Positions")
        )
        layout["signals"].update(
            Panel(self._render_signals_panel_rich(data.get("signals", [])), title="Signals")
        )
        layout["heatmap"].update(
            Panel(self._render_heatmap_panel_rich(data.get("sector_data", {})), title="Sector Heatmap")
        )
        layout["portfolio"].update(
            Panel(self._render_portfolio_panel_rich(data.get("portfolio", {})), title="Portfolio")
        )
        layout["risk"].update(
            Panel(self._render_risk_panel_rich(data.get("risk", {})), title="Risk")
        )
        layout["performance"].update(
            Panel(self._render_performance_panel_rich(data.get("performance", {})), title="Performance")
        )
        return layout

    # ------------------------------------------------------------------
    # Rich-specific panel renderers
    # ------------------------------------------------------------------
    def _render_scanner_panel_rich(self, data: dict) -> "Table":
        table = Table(show_header=True, expand=True)
        table.add_column("Ticker", style="bold")
        table.add_column("Price", justify="right")
        table.add_column("Chg%", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Sector")
        movers = data.get("top_movers", [])
        bottom = data.get("bottom_movers", [])
        vol_spikes = data.get("volume_spikes", [])
        combined = movers + bottom + vol_spikes
        for item in combined[:25]:
            chg = item.get("change_pct", 0)
            style = "green" if chg >= 0 else "red"
            table.add_row(
                item.get("ticker", ""),
                f"${item.get('price', 0):,.2f}",
                Text(f"{chg:+.2f}%", style=style),
                f"{item.get('volume', 0):,.0f}",
                item.get("sector", ""),
            )
        if not combined:
            table.add_row("--", "--", "--", "--", "Awaiting data...")
        return table

    def _render_positions_panel_rich(self, positions: list) -> "Table":
        table = Table(show_header=True, expand=True)
        table.add_column("Ticker", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Unreal PnL", justify="right")
        table.add_column("PnL%", justify="right")
        sorted_pos = sorted(positions, key=lambda p: abs(p.get("unrealized_pnl", 0)), reverse=True)
        for pos in sorted_pos[:20]:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            style = "green" if pnl >= 0 else "red"
            table.add_row(
                pos.get("ticker", ""),
                f"{pos.get('quantity', 0):,.0f}",
                f"${pos.get('avg_cost', 0):,.2f}",
                f"${pos.get('current_price', 0):,.2f}",
                Text(f"${pnl:+,.0f}", style=style),
                Text(f"{pnl_pct:+.2f}%", style=style),
            )
        if not positions:
            table.add_row("--", "--", "--", "--", "--", "No positions")
        return table

    def _render_signals_panel_rich(self, signals: list) -> "Table":
        table = Table(show_header=True, expand=True)
        table.add_column("Ticker", style="bold")
        table.add_column("Signal", justify="center")
        table.add_column("Votes", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Edge (bps)", justify="right")
        top = sorted(signals, key=lambda s: abs(s.get("vote_score", 0)), reverse=True)[:20]
        for sig in top:
            sig_type = sig.get("signal_type", "HOLD")
            style = "green" if sig_type == "BUY" else ("red" if sig_type == "SELL" else "yellow")
            table.add_row(
                sig.get("ticker", ""),
                Text(sig_type, style=style),
                f"{sig.get('vote_score', 0):.2f}",
                f"{sig.get('confidence', 0):.1%}",
                f"{sig.get('edge_bps', 0):+.1f}",
            )
        if not signals:
            table.add_row("--", "--", "--", "--", "Awaiting signals")
        return table

    def _render_heatmap_panel_rich(self, sector_data: dict) -> "Table":
        table = Table(show_header=True, expand=True)
        table.add_column("Sector", style="bold")
        table.add_column("1D", justify="right")
        table.add_column("1W", justify="right")
        table.add_column("1M", justify="right")
        for sector in GICS_SECTORS:
            info = sector_data.get(sector, {})
            d1 = info.get("1d", 0)
            w1 = info.get("1w", 0)
            m1 = info.get("1m", 0)
            table.add_row(
                sector,
                Text(f"{d1:+.2f}%", style="green" if d1 >= 0 else "red"),
                Text(f"{w1:+.2f}%", style="green" if w1 >= 0 else "red"),
                Text(f"{m1:+.2f}%", style="green" if m1 >= 0 else "red"),
            )
        return table

    def _render_portfolio_panel_rich(self, portfolio: dict) -> "Text":
        nav = portfolio.get("nav", 0)
        cash = portfolio.get("cash", 0)
        gross = portfolio.get("gross_exposure", 0)
        net = portfolio.get("net_exposure", 0)
        alloc = portfolio.get("sector_allocation", {})
        lines = [
            f"NAV:            ${nav:>14,.0f}",
            f"Cash:           ${cash:>14,.0f}",
            f"Gross Exposure: ${gross:>14,.0f}",
            f"Net Exposure:   ${net:>14,.0f}",
            "",
            "Sector Allocation:",
        ]
        for sec, pct in sorted(alloc.items(), key=lambda x: -x[1]):
            bar_len = int(pct * 30)
            lines.append(f"  {sec:<28s} {pct:5.1%} {'#' * bar_len}")
        return Text("\n".join(lines))

    def _render_risk_panel_rich(self, risk_data: dict) -> "Text":
        var95 = risk_data.get("var_95", 0)
        var99 = risk_data.get("var_99", 0)
        beta = risk_data.get("beta", 0)
        max_dd = risk_data.get("max_drawdown", 0)
        cur_dd = risk_data.get("current_drawdown", 0)
        gate = risk_data.get("risk_gate_status", "UNKNOWN")
        gate_style = "bold green" if gate == "OPEN" else "bold red"
        lines = Text()
        lines.append(f"VaR 95%:          {var95:+.2%}\n")
        lines.append(f"VaR 99%:          {var99:+.2%}\n")
        lines.append(f"Beta:             {beta:+.3f}\n")
        lines.append(f"Max Drawdown:     {max_dd:+.2%}\n")
        lines.append(f"Current Drawdown: {cur_dd:+.2%}\n")
        lines.append("\n")
        lines.append("Risk Gate: ", style="bold")
        lines.append(gate, style=gate_style)
        return lines

    def _render_performance_panel_rich(self, perf_data: dict) -> "Text":
        daily_pnl = perf_data.get("daily_pnl", 0)
        cum_ret = perf_data.get("cumulative_return", 0)
        sharpe = perf_data.get("sharpe", 0)
        win_rate = perf_data.get("win_rate", 0)
        history = perf_data.get("equity_curve", [])
        spark = _sparkline(history, width=40)
        pnl_style = "green" if daily_pnl >= 0 else "red"
        lines = Text()
        lines.append(f"Daily PnL:   ")
        lines.append(f"${daily_pnl:+,.0f}\n", style=pnl_style)
        lines.append(f"Cumul. Ret:  {cum_ret:+.2%}\n")
        lines.append(f"Sharpe:      {sharpe:.2f}\n")
        lines.append(f"Win Rate:    {win_rate:.1%}\n")
        lines.append(f"\nEquity: [{spark}]\n")
        return lines

    # ------------------------------------------------------------------
    # Plain-text panel renderers (return str)
    # ------------------------------------------------------------------
    def _render_scanner_panel(self, data: dict) -> str:
        """Real-time market scanner for 550+ symbol universe."""
        lines: List[str] = []
        header = f"{'Ticker':<8}{'Price':>10}{'Chg%':>8}{'Volume':>14}{'Sector':<20}"
        lines.append(f"{_BOLD}{_CYAN}{header}{_RESET}")
        lines.append("-" * len(header))
        movers = data.get("top_movers", [])
        bottom = data.get("bottom_movers", [])
        vol_spikes = data.get("volume_spikes", [])
        combined = movers + bottom + vol_spikes
        # Filter by thresholds (market cap, volume, price)
        mcap_min = data.get("mcap_min", 0)
        vol_min = data.get("vol_min", 0)
        price_min = data.get("price_min", 0)
        filtered = [
            item for item in combined
            if item.get("market_cap", float("inf")) >= mcap_min
            and item.get("volume", 0) >= vol_min
            and item.get("price", 0) >= price_min
        ]
        for item in filtered[:25]:
            chg = item.get("change_pct", 0)
            color = _GREEN if chg >= 0 else _RED
            lines.append(
                f"{item.get('ticker', ''):<8}"
                f"${item.get('price', 0):>9,.2f}"
                f"{color}{chg:>+7.2f}%{_RESET}"
                f"{item.get('volume', 0):>14,.0f}"
                f"  {item.get('sector', ''):<20}"
            )
        if not filtered:
            lines.append(f"  {_DIM}Awaiting scanner data...{_RESET}")
        # Sector breakdown summary
        sector_counts: Dict[str, int] = {}
        for item in combined:
            sec = item.get("sector", "Other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if sector_counts:
            lines.append("")
            lines.append(f"{_BOLD}Sector breakdown:{_RESET}")
            for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {sec:<24s} {cnt}")
        return "\n".join(lines)

    def _render_positions_panel(self, positions: list) -> str:
        """Current portfolio positions sorted by absolute PnL."""
        lines: List[str] = []
        header = f"{'Ticker':<8}{'Qty':>8}{'AvgCost':>10}{'Price':>10}{'Unreal PnL':>12}{'PnL%':>8}"
        lines.append(f"{_BOLD}{_CYAN}{header}{_RESET}")
        lines.append("-" * len(header))
        sorted_pos = sorted(positions, key=lambda p: abs(p.get("unrealized_pnl", 0)), reverse=True)
        for pos in sorted_pos[:20]:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            color = _GREEN if pnl >= 0 else _RED
            lines.append(
                f"{pos.get('ticker', ''):<8}"
                f"{pos.get('quantity', 0):>8,.0f}"
                f"${pos.get('avg_cost', 0):>9,.2f}"
                f"${pos.get('current_price', 0):>9,.2f}"
                f"{color}${pnl:>+11,.0f}{_RESET}"
                f"{color}{pnl_pct:>+7.2f}%{_RESET}"
            )
        if not positions:
            lines.append(f"  {_DIM}No open positions.{_RESET}")
        return "\n".join(lines)

    def _render_signals_panel(self, signals: list) -> str:
        """Latest signals from ML ensemble — top 20 strongest."""
        lines: List[str] = []
        header = f"{'Ticker':<8}{'Signal':<8}{'Votes':>8}{'Conf':>8}{'Edge(bps)':>10}"
        lines.append(f"{_BOLD}{_CYAN}{header}{_RESET}")
        lines.append("-" * len(header))
        top = sorted(signals, key=lambda s: abs(s.get("vote_score", 0)), reverse=True)[:20]
        for sig in top:
            sig_type = sig.get("signal_type", "HOLD")
            if sig_type == "BUY":
                color = _GREEN
            elif sig_type == "SELL":
                color = _RED
            else:
                color = _YELLOW
            lines.append(
                f"{sig.get('ticker', ''):<8}"
                f"{color}{sig_type:<8}{_RESET}"
                f"{sig.get('vote_score', 0):>8.2f}"
                f"{sig.get('confidence', 0):>7.1%}"
                f"{sig.get('edge_bps', 0):>+10.1f}"
            )
        if not signals:
            lines.append(f"  {_DIM}Awaiting signal generation...{_RESET}")
        return "\n".join(lines)

    def _render_heatmap_panel(self, sector_data: dict) -> str:
        """Sector performance heatmap — 11 GICS sectors with color coding."""
        lines: List[str] = []
        header = f"{'Sector':<28}{'1D':>8}{'1W':>8}{'1M':>8}"
        lines.append(f"{_BOLD}{_CYAN}{header}{_RESET}")
        lines.append("-" * len(header))
        for sector in GICS_SECTORS:
            info = sector_data.get(sector, {})
            d1 = info.get("1d", 0)
            w1 = info.get("1w", 0)
            m1 = info.get("1m", 0)
            c1 = _GREEN if d1 >= 0 else _RED
            c2 = _GREEN if w1 >= 0 else _RED
            c3 = _GREEN if m1 >= 0 else _RED
            lines.append(
                f"{sector:<28}"
                f"{c1}{d1:>+7.2f}%{_RESET}"
                f"{c2}{w1:>+7.2f}%{_RESET}"
                f"{c3}{m1:>+7.2f}%{_RESET}"
            )
        return "\n".join(lines)

    def _render_portfolio_panel(self, portfolio: dict) -> str:
        """Portfolio summary — NAV, cash, exposure, sector allocation."""
        nav = portfolio.get("nav", 0)
        cash = portfolio.get("cash", 0)
        gross = portfolio.get("gross_exposure", 0)
        net = portfolio.get("net_exposure", 0)
        alloc = portfolio.get("sector_allocation", {})
        lines = [
            f"{_BOLD}Portfolio Summary{_RESET}",
            f"  NAV:            ${nav:>14,.0f}",
            f"  Cash:           ${cash:>14,.0f}",
            f"  Gross Exposure: ${gross:>14,.0f}",
            f"  Net Exposure:   ${net:>14,.0f}",
        ]
        if alloc:
            lines.append("")
            lines.append(f"  {_BOLD}Sector Allocation:{_RESET}")
            for sec, pct in sorted(alloc.items(), key=lambda x: -x[1]):
                bar = "#" * int(pct * 30)
                lines.append(f"    {sec:<26s} {pct:5.1%} {_CYAN}{bar}{_RESET}")
        return "\n".join(lines)

    def _render_risk_panel(self, risk_data: dict) -> str:
        """Risk metrics — VaR, beta, drawdown, risk gate status."""
        var95 = risk_data.get("var_95", 0)
        var99 = risk_data.get("var_99", 0)
        beta = risk_data.get("beta", 0)
        max_dd = risk_data.get("max_drawdown", 0)
        cur_dd = risk_data.get("current_drawdown", 0)
        gate = risk_data.get("risk_gate_status", "UNKNOWN")
        gate_color = _GREEN if gate == "OPEN" else _RED
        lines = [
            f"{_BOLD}Risk Metrics{_RESET}",
            f"  VaR 95%:          {var95:+.2%}",
            f"  VaR 99%:          {var99:+.2%}",
            f"  Beta:             {beta:+.3f}",
            f"  Max Drawdown:     {max_dd:+.2%}",
            f"  Current Drawdown: {cur_dd:+.2%}",
            "",
            f"  Risk Gate: {gate_color}{_BOLD}{gate}{_RESET}",
        ]
        return "\n".join(lines)

    def _render_performance_panel(self, perf_data: dict) -> str:
        """Performance chart — daily PnL, returns, Sharpe, win rate, sparkline."""
        daily_pnl = perf_data.get("daily_pnl", 0)
        cum_ret = perf_data.get("cumulative_return", 0)
        sharpe = perf_data.get("sharpe", 0)
        win_rate = perf_data.get("win_rate", 0)
        history = perf_data.get("equity_curve", [])
        spark = _sparkline(history, width=40)
        pnl_color = _GREEN if daily_pnl >= 0 else _RED
        lines = [
            f"{_BOLD}Performance{_RESET}",
            f"  Daily PnL:    {pnl_color}${daily_pnl:>+12,.0f}{_RESET}",
            f"  Cumul. Ret:   {cum_ret:>+12.2%}",
            f"  Sharpe:       {sharpe:>12.2f}",
            f"  Win Rate:     {win_rate:>11.1%}",
            "",
            f"  Equity: [{_CYAN}{spark}{_RESET}]",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full ASCII dashboard composition
    # ------------------------------------------------------------------
    def format_full_dashboard(self, engine_data: dict) -> str:
        """Render all panels into one large ASCII dashboard string.

        Layout
        ------
        Header:       timestamp | NAV | regime
        Upper row:    scanner (left)   |  positions (right)
        Full-width:   signals
        Full-width:   heatmap
        Footer row:   portfolio | risk | performance
        """
        try:
            return self._compose_dashboard(engine_data)
        except Exception:
            logger.exception("Failed to compose dashboard")
            return "[Dashboard render error — check logs]"

    def _compose_dashboard(self, data: dict) -> str:
        width = 120
        sep = "=" * width

        # -- Header --
        nav = data.get("portfolio", {}).get("nav", 0)
        regime = data.get("regime", "UNKNOWN")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"{_BOLD}{_CYAN}"
            f"{'METADRON CAPITAL':^{width}}\n"
            f"{ts}  |  NAV: ${nav:,.0f}  |  Regime: {regime}  |  Universe: {len(SCANNER_UNIVERSE)} symbols"
            f"{_RESET}"
        )

        # -- Scanner + Positions side-by-side --
        scanner_text = self._render_scanner_panel(data.get("scanner", {}))
        positions_text = self._render_positions_panel(data.get("positions", []))
        upper = self._side_by_side(scanner_text, positions_text, width)

        # -- Full-width signals --
        signals_text = self._render_signals_panel(data.get("signals", []))

        # -- Full-width heatmap --
        heatmap_text = self._render_heatmap_panel(data.get("sector_data", {}))

        # -- Footer: portfolio | risk | performance --
        portfolio_text = self._render_portfolio_panel(data.get("portfolio", {}))
        risk_text = self._render_risk_panel(data.get("risk", {}))
        perf_text = self._render_performance_panel(data.get("performance", {}))
        footer = self._three_columns(portfolio_text, risk_text, perf_text, width)

        parts = [
            sep,
            header,
            sep,
            f"{_BOLD} SCANNER{' ' * 50}POSITIONS{_RESET}",
            upper,
            sep,
            f"{_BOLD} SIGNALS{_RESET}",
            signals_text,
            sep,
            f"{_BOLD} SECTOR HEATMAP{_RESET}",
            heatmap_text,
            sep,
            footer,
            sep,
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _side_by_side(left: str, right: str, total_width: int) -> str:
        """Merge two text blocks into a 2-column layout."""
        import re
        col_w = total_width // 2 - 2
        left_lines = left.split("\n")
        right_lines = right.split("\n")
        max_rows = max(len(left_lines), len(right_lines))
        out: List[str] = []
        for i in range(max_rows):
            l = left_lines[i] if i < len(left_lines) else ""
            r = right_lines[i] if i < len(right_lines) else ""
            # Compute visible length ignoring ANSI
            l_vis = len(re.sub(r"\033\[[0-9;]*m", "", l))
            pad = max(0, col_w - l_vis)
            out.append(l + " " * pad + " | " + r)
        return "\n".join(out)

    @staticmethod
    def _three_columns(c1: str, c2: str, c3: str, total_width: int) -> str:
        """Merge three text blocks into a 3-column layout."""
        import re
        col_w = total_width // 3 - 2
        lines1 = c1.split("\n")
        lines2 = c2.split("\n")
        lines3 = c3.split("\n")
        max_rows = max(len(lines1), len(lines2), len(lines3))
        out: List[str] = []
        for i in range(max_rows):
            parts = []
            for lines in (lines1, lines2, lines3):
                t = lines[i] if i < len(lines) else ""
                vis = len(re.sub(r"\033\[[0-9;]*m", "", t))
                parts.append(t + " " * max(0, col_w - vis))
            out.append(" | ".join(parts))
        return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────────────────────────────
_default_dashboard: Optional[LiveDashboard] = None


def get_dashboard(refresh_interval: float = 5.0) -> LiveDashboard:
    """Return (and lazily create) the singleton dashboard instance."""
    global _default_dashboard
    if _default_dashboard is None:
        _default_dashboard = LiveDashboard(refresh_interval=refresh_interval)
    return _default_dashboard


# ──────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO)

    sample_data = {
        "regime": "RISK_ON",
        "scanner": {
            "top_movers": [
                {"ticker": t, "price": random.uniform(20, 500),
                 "change_pct": random.uniform(1, 12), "volume": random.randint(1_000_000, 50_000_000),
                 "sector": random.choice(GICS_SECTORS), "market_cap": 1e10}
                for t in random.sample(SCANNER_UNIVERSE, 10)
            ],
            "bottom_movers": [
                {"ticker": t, "price": random.uniform(20, 500),
                 "change_pct": random.uniform(-12, -1), "volume": random.randint(1_000_000, 50_000_000),
                 "sector": random.choice(GICS_SECTORS), "market_cap": 1e10}
                for t in random.sample(SCANNER_UNIVERSE, 10)
            ],
            "volume_spikes": [
                {"ticker": t, "price": random.uniform(20, 500),
                 "change_pct": random.uniform(-5, 5), "volume": random.randint(20_000_000, 100_000_000),
                 "sector": random.choice(GICS_SECTORS), "market_cap": 1e10}
                for t in random.sample(SCANNER_UNIVERSE, 5)
            ],
        },
        "positions": [
            {"ticker": t, "quantity": random.randint(100, 5000),
             "avg_cost": (ac := random.uniform(50, 400)),
             "current_price": ac * random.uniform(0.9, 1.15),
             "unrealized_pnl": random.uniform(-15000, 25000),
             "pnl_pct": random.uniform(-8, 12)}
            for t in random.sample(SCANNER_UNIVERSE, 12)
        ],
        "signals": [
            {"ticker": t, "signal_type": random.choice(["BUY", "SELL", "HOLD"]),
             "vote_score": random.uniform(-1, 1), "confidence": random.uniform(0.4, 0.95),
             "edge_bps": random.uniform(-50, 80)}
            for t in random.sample(SCANNER_UNIVERSE, 25)
        ],
        "sector_data": {
            sec: {"1d": random.uniform(-3, 3), "1w": random.uniform(-5, 5),
                  "1m": random.uniform(-8, 10)}
            for sec in GICS_SECTORS
        },
        "portfolio": {
            "nav": 10_250_000, "cash": 2_150_000,
            "gross_exposure": 8_100_000, "net_exposure": 4_300_000,
            "sector_allocation": {sec: random.uniform(0.03, 0.18) for sec in GICS_SECTORS},
        },
        "risk": {
            "var_95": -0.018, "var_99": -0.032, "beta": 0.72,
            "max_drawdown": -0.085, "current_drawdown": -0.023,
            "risk_gate_status": "OPEN",
        },
        "performance": {
            "daily_pnl": 34_500, "cumulative_return": 0.127,
            "sharpe": 1.85, "win_rate": 0.583,
            "equity_curve": [10_000_000 + random.randint(-50_000, 80_000) * i for i in range(60)],
        },
    }

    dash = LiveDashboard(refresh_interval=2.0)
    print(dash.format_full_dashboard(sample_data))
    print(f"\nUniverse size: {len(SCANNER_UNIVERSE)} symbols")
