"""LiveEarningsGraph — Live P&L visualization with benchmark comparison.

Provides:
    - ASCII line chart showing NAV over time with benchmark overlay
    - Daily P&L bar chart
    - Current NAV, peak NAV, drawdown, daily PnL stats
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("numpy not available — LiveEarningsGraph degraded")

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_BLUE = "\033[94m"
ANSI_WHITE = "\033[97m"


class LiveEarningsGraph:
    """Live P&L visualization with benchmark comparison."""

    def __init__(self, benchmark_ticker: str = "SPY"):
        self.benchmark_ticker = benchmark_ticker
        self._nav_history: List[Tuple[str, float]] = []  # (timestamp, nav)
        self._benchmark_history: List[Tuple[str, float]] = []
        self._peak_nav: float = 0.0
        self._initial_nav: Optional[float] = None
        logger.info("LiveEarningsGraph initialized — benchmark=%s", benchmark_ticker)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, nav: float, timestamp: str = None) -> None:
        """Record a new NAV observation.

        Parameters
        ----------
        nav : float
            Current portfolio net asset value.
        timestamp : str, optional
            ISO-format timestamp. Defaults to now.
        """
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._nav_history.append((timestamp, float(nav)))
            if self._initial_nav is None:
                self._initial_nav = float(nav)
            if nav > self._peak_nav:
                self._peak_nav = float(nav)
            logger.debug("NAV update: %.2f at %s", nav, timestamp)
        except Exception as e:
            logger.error("LiveEarningsGraph.update failed: %s", e)

    # ------------------------------------------------------------------
    # ASCII line chart
    # ------------------------------------------------------------------
    def generate_ascii_chart(self, width: int = 80, height: int = 20) -> str:
        """Generate ASCII line chart showing NAV over time.

        Includes benchmark overlay, current NAV, peak NAV, drawdown, daily PnL.
        """
        try:
            if not self._nav_history:
                return "[LiveEarningsGraph] No NAV data — call update() first"

            navs = [v for _, v in self._nav_history]
            n = len(navs)

            # Compute stats
            current_nav = navs[-1]
            peak = self._peak_nav
            drawdown = ((current_nav - peak) / peak * 100) if peak > 0 else 0.0
            daily_pnl = navs[-1] - navs[-2] if n >= 2 else 0.0
            total_return = ((current_nav / self._initial_nav - 1) * 100) if self._initial_nav else 0.0

            # Resample to fit width
            plot_width = width - 12  # leave room for y-axis labels
            if n > plot_width:
                step = n / plot_width
                sampled = [navs[int(i * step)] for i in range(plot_width)]
            else:
                sampled = navs
                plot_width = len(sampled)

            y_min = min(sampled) * 0.998
            y_max = max(sampled) * 1.002
            y_range = y_max - y_min if y_max > y_min else 1.0

            # Build grid
            grid = [[" " for _ in range(plot_width)] for _ in range(height)]

            for x, val in enumerate(sampled):
                y = int((val - y_min) / y_range * (height - 1))
                y = max(0, min(height - 1, y))
                row = height - 1 - y
                color = ANSI_GREEN if val >= (self._initial_nav or val) else ANSI_RED
                grid[row][x] = f"{color}*{ANSI_RESET}"

            # Render
            lines = []
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * width}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}  LIVE P&L CHART  —  Benchmark: {self.benchmark_ticker}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * width}{ANSI_RESET}")

            for r in range(height):
                y_val = y_max - (r / (height - 1)) * y_range if height > 1 else y_max
                label = f"${y_val:>9,.0f}" if y_val >= 1000 else f" {y_val:>9.2f}"
                row_str = label + " |" + "".join(grid[r])
                lines.append(row_str)

            # X-axis
            x_axis = " " * 11 + "+" + "-" * plot_width
            lines.append(x_axis)

            # Stats block
            lines.append("")
            dd_color = ANSI_RED if drawdown < 0 else ANSI_GREEN
            pnl_color = ANSI_GREEN if daily_pnl >= 0 else ANSI_RED
            lines.append(
                f"  Current NAV: {ANSI_BOLD}${current_nav:,.2f}{ANSI_RESET}  |  "
                f"Peak: ${peak:,.2f}  |  "
                f"Drawdown: {dd_color}{drawdown:+.2f}%{ANSI_RESET}  |  "
                f"Daily PnL: {pnl_color}${daily_pnl:+,.2f}{ANSI_RESET}"
            )
            ret_color = ANSI_GREEN if total_return >= 0 else ANSI_RED
            lines.append(
                f"  Total Return: {ret_color}{total_return:+.2f}%{ANSI_RESET}  |  "
                f"Observations: {n}"
            )
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * width}{ANSI_RESET}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("generate_ascii_chart failed: %s", e)
            return f"[LiveEarningsGraph] Chart error: {e}"

    # ------------------------------------------------------------------
    # Daily P&L bar chart
    # ------------------------------------------------------------------
    def generate_pnl_bar_chart(self, daily_pnls: list = None, width: int = 60) -> str:
        """Generate daily P&L bar chart.

        Parameters
        ----------
        daily_pnls : list, optional
            List of daily P&L floats. If None, computed from NAV history.
        width : int
            Chart width in characters.
        """
        try:
            if daily_pnls is None:
                if len(self._nav_history) < 2:
                    return "[LiveEarningsGraph] Need >= 2 NAV observations for PnL bars"
                navs = [v for _, v in self._nav_history]
                daily_pnls = [navs[i] - navs[i - 1] for i in range(1, len(navs))]

            if not daily_pnls:
                return "[LiveEarningsGraph] No P&L data"

            max_abs = max(abs(p) for p in daily_pnls) if daily_pnls else 1.0
            bar_width = (width - 20) // 2  # half for positive, half for negative

            lines = []
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * width}{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}  DAILY P&L BAR CHART{ANSI_RESET}")
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * width}{ANSI_RESET}")

            for i, pnl in enumerate(daily_pnls):
                bar_len = int(abs(pnl) / max_abs * bar_width) if max_abs > 0 else 0
                bar_len = max(bar_len, 1)

                if pnl >= 0:
                    pad = " " * bar_width
                    bar = f"{ANSI_GREEN}{'#' * bar_len}{ANSI_RESET}"
                    line = f"  D{i + 1:>3} {pad}|{bar} ${pnl:+,.0f}"
                else:
                    pad_right = " " * (bar_width - bar_len)
                    bar = f"{ANSI_RED}{'#' * bar_len}{ANSI_RESET}"
                    line = f"  D{i + 1:>3} {pad_right}{bar}|   ${pnl:+,.0f}"
                lines.append(line)

            # Summary
            total = sum(daily_pnls)
            wins = sum(1 for p in daily_pnls if p > 0)
            losses = sum(1 for p in daily_pnls if p < 0)
            win_rate = wins / len(daily_pnls) * 100 if daily_pnls else 0
            lines.append("")
            t_color = ANSI_GREEN if total >= 0 else ANSI_RED
            lines.append(
                f"  Total: {t_color}${total:+,.2f}{ANSI_RESET}  |  "
                f"Win/Loss: {wins}/{losses}  |  "
                f"Win Rate: {win_rate:.0f}%"
            )
            lines.append(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * width}{ANSI_RESET}")
            return "\n".join(lines)
        except Exception as e:
            logger.error("generate_pnl_bar_chart failed: %s", e)
            return f"[LiveEarningsGraph] PnL bar chart error: {e}"
