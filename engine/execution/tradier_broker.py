"""TradierBroker — Live execution via Tradier Brokerage API.

Drop-in replacement for PaperBroker. Implements the same interface so
ExecutionEngine can swap between paper and live with a single config toggle.

Environment variables:
    TRADIER_API_KEY       — Bearer token (sandbox or production)
    TRADIER_ACCOUNT_ID    — Brokerage account ID
    TRADIER_ENVIRONMENT   — "sandbox" (default) or "production"

Tradier API reference: https://docs.tradier.com/

Endpoints used:
    POST   /v1/accounts/{id}/orders          — place equity order
    PUT    /v1/accounts/{id}/orders/{oid}     — modify order
    DELETE /v1/accounts/{id}/orders/{oid}     — cancel order
    GET    /v1/accounts/{id}/orders           — list orders
    GET    /v1/accounts/{id}/orders/{oid}     — single order
    GET    /v1/accounts/{id}/positions        — positions
    GET    /v1/accounts/{id}/balances         — balances (cash, NAV)
    GET    /v1/accounts/{id}/gainloss         — realized P&L
    GET    /v1/markets/quotes                 — real-time quotes
"""

import os
import json
import time
import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum

import httpx
import numpy as np
import pandas as pd

from .paper_broker import (
    Order, OrderSide, OrderType, OrderStatus, SignalType,
    Position, PortfolioState, RiskLimiter, PerformanceTracker,
    DailyTargetManager, LiveDashboardState, RiskProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_URLS = {
    "sandbox": "https://sandbox.tradier.com",
    "production": "https://api.tradier.com",
}

# Tradier side → Metadron OrderSide mapping
_TRADIER_SIDE_MAP = {
    "buy": OrderSide.BUY,
    "sell": OrderSide.SELL,
    "sell_short": OrderSide.SHORT,
    "buy_to_cover": OrderSide.COVER,
}

_SIDE_TO_TRADIER = {
    OrderSide.BUY: "buy",
    OrderSide.SELL: "sell",
    OrderSide.SHORT: "sell_short",
    OrderSide.COVER: "buy_to_cover",
}

_TYPE_TO_TRADIER = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
}

_TRADIER_STATUS_MAP = {
    "pending": OrderStatus.PENDING,
    "open": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PENDING,
    "filled": OrderStatus.FILLED,
    "expired": OrderStatus.CANCELLED,
    "canceled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


# ---------------------------------------------------------------------------
# Tradier API Client (low-level)
# ---------------------------------------------------------------------------
class TradierAPIClient:
    """Low-level HTTP client for the Tradier Brokerage API."""

    MAX_RETRIES = 4
    BACKOFF_BASE = 2  # seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("TRADIER_API_KEY", "")
        self.account_id = account_id or os.environ.get("TRADIER_ACCOUNT_ID", "")
        env = (environment or os.environ.get("TRADIER_ENVIRONMENT", "sandbox")).lower()
        if env not in _BASE_URLS:
            raise ValueError(f"Invalid environment '{env}'. Use 'sandbox' or 'production'.")
        self.environment = env
        self.base_url = _BASE_URLS[env]
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            timeout=15.0,
        )

    @property
    def _account_url(self) -> str:
        return f"{self.base_url}/v1/accounts/{self.account_id}"

    def _request(self, method: str, url: str, **kwargs) -> dict:
        """Execute an HTTP request with retry + exponential backoff."""
        last_exc = None
        resp = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectError as e:
                last_exc = e
                wait = self.BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Tradier request failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, self.MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    logger.error("Tradier API error %d: %s",
                                 e.response.status_code, e.response.text)
                    raise
                last_exc = e
                wait = self.BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Tradier server error %d (attempt %d/%d), retrying in %ds",
                    e.response.status_code, attempt + 1, self.MAX_RETRIES, wait,
                )
                time.sleep(wait)
        raise ConnectionError(
            f"Tradier API request failed after {self.MAX_RETRIES} retries"
        ) from last_exc

    # --- Account endpoints ---------------------------------------------------

    def get_balances(self) -> dict:
        """GET /v1/accounts/{id}/balances"""
        data = self._request("GET", f"{self._account_url}/balances")
        return data.get("balances", data)

    def get_positions(self) -> list[dict]:
        """GET /v1/accounts/{id}/positions"""
        data = self._request("GET", f"{self._account_url}/positions")
        positions = data.get("positions", {})
        if positions == "null" or positions is None:
            return []
        pos_list = positions.get("position", [])
        if isinstance(pos_list, dict):
            return [pos_list]
        return pos_list

    def get_orders(self) -> list[dict]:
        """GET /v1/accounts/{id}/orders"""
        data = self._request("GET", f"{self._account_url}/orders")
        orders = data.get("orders", {})
        if orders == "null" or orders is None:
            return []
        order_list = orders.get("order", [])
        if isinstance(order_list, dict):
            return [order_list]
        return order_list

    def get_order(self, order_id: str) -> dict:
        """GET /v1/accounts/{id}/orders/{order_id}"""
        data = self._request("GET", f"{self._account_url}/orders/{order_id}")
        return data.get("order", data)

    def get_gainloss(self) -> list[dict]:
        """GET /v1/accounts/{id}/gainloss"""
        data = self._request("GET", f"{self._account_url}/gainloss")
        gl = data.get("gainloss", {})
        if gl == "null" or gl is None:
            return []
        entries = gl.get("closed_position", [])
        if isinstance(entries, dict):
            return [entries]
        return entries

    # --- Trading endpoints ---------------------------------------------------

    def place_equity_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        duration: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        preview: bool = False,
    ) -> dict:
        """POST /v1/accounts/{id}/orders — Place an equity order.

        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            side: "buy", "sell", "sell_short", "buy_to_cover"
            quantity: Number of shares
            order_type: "market", "limit", "stop", "stop_limit"
            duration: "day", "gtc", "pre", "post"
            limit_price: Required for limit/stop_limit orders
            stop_price: Required for stop/stop_limit orders
            preview: If True, validate without executing
        """
        payload = {
            "class": "equity",
            "symbol": symbol,
            "side": side,
            "quantity": str(quantity),
            "type": order_type,
            "duration": duration,
        }
        if limit_price is not None:
            payload["price"] = str(limit_price)
        if stop_price is not None:
            payload["stop"] = str(stop_price)
        if preview:
            payload["preview"] = "true"

        data = self._request(
            "POST",
            f"{self._account_url}/orders",
            data=payload,
        )
        return data.get("order", data)

    def modify_order(
        self,
        order_id: str,
        order_type: Optional[str] = None,
        duration: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> dict:
        """PUT /v1/accounts/{id}/orders/{order_id}"""
        payload = {}
        if order_type:
            payload["type"] = order_type
        if duration:
            payload["duration"] = duration
        if limit_price is not None:
            payload["price"] = str(limit_price)
        if stop_price is not None:
            payload["stop"] = str(stop_price)
        data = self._request(
            "PUT",
            f"{self._account_url}/orders/{order_id}",
            data=payload,
        )
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> dict:
        """DELETE /v1/accounts/{id}/orders/{order_id}"""
        data = self._request(
            "DELETE",
            f"{self._account_url}/orders/{order_id}",
        )
        return data

    # --- Market data endpoints -----------------------------------------------

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """GET /v1/markets/quotes — Real-time quotes.

        Args:
            symbols: List of ticker symbols
        """
        if not symbols:
            return []
        data = self._request(
            "GET",
            f"{self.base_url}/v1/markets/quotes",
            params={"symbols": ",".join(symbols), "greeks": "false"},
        )
        quotes = data.get("quotes", {})
        if quotes == "null" or quotes is None:
            return []
        quote_list = quotes.get("quote", [])
        if isinstance(quote_list, dict):
            return [quote_list]
        return quote_list

    def get_quote(self, symbol: str) -> dict:
        """Get a single quote. Returns empty dict if unavailable."""
        quotes = self.get_quotes([symbol])
        return quotes[0] if quotes else {}


# ---------------------------------------------------------------------------
# TradierBroker — Drop-in replacement for PaperBroker
# ---------------------------------------------------------------------------
class TradierBroker:
    """Live execution broker via Tradier API.

    Implements the same interface as PaperBroker so ExecutionEngine can
    swap between paper and live with zero code changes.

    In sandbox mode, all orders execute against Tradier's paper trading
    engine with delayed market data. In production mode, orders are real.
    """

    def __init__(
        self,
        initial_cash: float = 1_000.0,
        log_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        environment: Optional[str] = None,
        enable_risk_limits: bool = True,
        daily_target_pct: float = 0.05,
    ):
        # Tradier API client
        self.client = TradierAPIClient(
            api_key=api_key,
            account_id=account_id,
            environment=environment,
        )

        # Portfolio state — synced from Tradier on each call
        self.state = PortfolioState(cash=initial_cash, nav=initial_cash)

        # Logging
        self.log_dir = log_dir or Path("logs/tradier_broker")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._orders: list[Order] = []
        self._trade_log: list[dict] = []

        # Risk & performance (reuse from PaperBroker)
        self._initial_cash = initial_cash
        self._risk_limiter = RiskLimiter() if enable_risk_limits else None
        self._perf_tracker = PerformanceTracker(initial_nav=initial_cash)
        self._daily_pnl_today: float = 0.0
        self._last_eod_date: Optional[str] = None
        self._eod_nav_history: list[tuple[str, float]] = []

        # Daily target manager — 5% compound daily minimum
        self._target_manager = DailyTargetManager(initial_nav=initial_cash)
        self._target_manager.DAILY_TARGET_PCT = daily_target_pct

        # Live dashboard state emitter
        self._dashboard = LiveDashboardState()

        # Order ID mapping: local_id → tradier_order_id
        self._order_id_map: dict[str, str] = {}

        # Quote cache: ticker → (timestamp, price)
        self._quote_cache: dict[str, tuple[float, float]] = {}
        self._quote_cache_ttl = 5.0  # seconds

        # Sync initial state from Tradier
        try:
            self._sync_balances()
            self._sync_positions()
            logger.info(
                "TradierBroker initialized (%s): cash=$%.2f, %d positions",
                self.client.environment, self.state.cash,
                len(self.state.positions),
            )
        except Exception as e:
            logger.warning("TradierBroker init sync failed (offline?): %s", e)

    # --- Tradier state sync --------------------------------------------------

    def _sync_balances(self):
        """Sync cash and NAV from Tradier account balances."""
        balances = self.client.get_balances()
        # Tradier returns different keys for margin vs cash accounts
        self.state.cash = float(
            balances.get("cash", {}).get("cash_available", 0)
            or balances.get("total_cash", 0)
            or balances.get("cash_available", 0)
        )
        self.state.nav = float(
            balances.get("total_equity", 0)
            or balances.get("account_value", 0)
            or self.state.cash
        )

    def _sync_positions(self):
        """Sync positions from Tradier."""
        tradier_positions = self.client.get_positions()
        synced: dict[str, Position] = {}
        for tp in tradier_positions:
            ticker = tp.get("symbol", "")
            if not ticker:
                continue
            qty = int(tp.get("quantity", 0))
            cost_basis = float(tp.get("cost_basis", 0))
            avg_cost = cost_basis / abs(qty) if qty != 0 else 0.0
            current_price = self._get_current_price(ticker)
            unrealized = (current_price - avg_cost) * qty if qty != 0 else 0.0
            synced[ticker] = Position(
                ticker=ticker,
                quantity=qty,
                avg_cost=avg_cost,
                current_price=current_price,
                unrealized_pnl=unrealized,
                realized_pnl=0.0,  # tracked via gainloss endpoint
            )
        self.state.positions = synced

    # --- Pre-trade risk checks (same as PaperBroker) -------------------------

    def _check_risk_limits(self, ticker: str, side: OrderSide,
                           quantity: int, price: float) -> tuple[bool, str]:
        if self._risk_limiter is None:
            return True, ""
        order_value = quantity * price
        nav = self.state.nav if self.state.nav > 0 else self._initial_cash
        sector = ""
        if ticker in self.state.positions:
            sector = self.state.positions[ticker].sector
        exposures = self.compute_exposures()
        gross = exposures.get("gross", 0.0)
        net = exposures.get("net", 0.0)
        passed, failures = self._risk_limiter.run_all_checks(
            order_value=order_value, ticker=ticker, sector=sector,
            positions=self.state.positions, nav=nav,
            daily_pnl=self._daily_pnl_today,
            gross_exposure=gross, net_exposure=net,
        )
        if not passed:
            return False, "; ".join(failures)
        return True, ""

    # --- Order execution (LIVE via Tradier API) ------------------------------

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        reason: str = "",
    ) -> Order:
        """Place a live order via Tradier API.

        Maintains the same signature as PaperBroker.place_order().
        """
        local_id = str(uuid.uuid4())[:8]
        order = Order(
            id=local_id,
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
            quantity=quantity,
            limit_price=limit_price,
            signal_type=signal_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )

        # Get current price for risk checks
        price = self._get_current_price(ticker)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = f"No price data for {ticker}"
            self._orders.append(order)
            return order

        # Update daily target manager
        nav = self.compute_nav()
        self._target_manager.update(nav)

        # Check if new positions allowed under current risk profile
        if side in (OrderSide.BUY, OrderSide.SHORT):
            if not self._target_manager.allow_new_positions():
                order.status = OrderStatus.REJECTED
                order.reason = (
                    f"DEFENSIVE mode — daily target exceeded "
                    f"({self._target_manager.profile.value}), no new positions"
                )
                self._orders.append(order)
                return order

            # Apply position size multiplier from risk profile
            pos_mult = self._target_manager.get_position_multiplier()
            if pos_mult < 1.0:
                quantity = max(1, int(quantity * pos_mult))
                order.quantity = quantity

        # Pre-trade risk checks
        risk_ok, risk_reason = self._check_risk_limits(ticker, side, quantity, price)
        if not risk_ok:
            order.status = OrderStatus.REJECTED
            order.reason = f"Risk limit: {risk_reason}"
            self._orders.append(order)
            return order

        # --- Submit to Tradier API ---
        tradier_side = _SIDE_TO_TRADIER[side]
        tradier_type = _TYPE_TO_TRADIER.get(order.order_type, "market")

        try:
            result = self.client.place_equity_order(
                symbol=ticker,
                side=tradier_side,
                quantity=quantity,
                order_type=tradier_type,
                limit_price=limit_price,
                duration="day",
            )

            tradier_order_id = str(result.get("id", ""))
            if not tradier_order_id:
                order.status = OrderStatus.REJECTED
                order.reason = f"Tradier returned no order ID: {result}"
                self._orders.append(order)
                return order

            self._order_id_map[local_id] = tradier_order_id
            order.status = OrderStatus.PENDING

            # Poll for fill (market orders fill near-instantly on Tradier)
            filled_order = self._poll_order_fill(tradier_order_id, timeout=10.0)
            if filled_order:
                order.fill_price = float(filled_order.get("avg_fill_price", price))
                order.fill_timestamp = (
                    filled_order.get("transaction_date", "")
                    or datetime.now(timezone.utc).isoformat()
                )
                status_str = filled_order.get("status", "pending").lower()
                order.status = _TRADIER_STATUS_MAP.get(status_str, OrderStatus.PENDING)
            else:
                # Market orders: assume filled at current price if poll timed out
                if order.order_type == OrderType.MARKET:
                    order.fill_price = price
                    order.fill_timestamp = datetime.now(timezone.utc).isoformat()
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PENDING

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = f"Tradier API error: {e}"
            logger.error("Tradier place_order failed for %s: %s", ticker, e)
            self._orders.append(order)
            return order

        # If filled, sync state
        if order.status == OrderStatus.FILLED:
            self._sync_after_fill(order)

        self._orders.append(order)
        self._log_trade(order)
        return order

    def _poll_order_fill(self, tradier_order_id: str,
                         timeout: float = 10.0) -> Optional[dict]:
        """Poll Tradier for order fill status. Returns order dict or None."""
        start = time.time()
        interval = 0.5
        while time.time() - start < timeout:
            try:
                order_data = self.client.get_order(tradier_order_id)
                status = order_data.get("status", "").lower()
                if status in ("filled", "rejected", "canceled", "expired"):
                    return order_data
            except Exception as e:
                logger.debug("Poll error for order %s: %s", tradier_order_id, e)
            time.sleep(interval)
            interval = min(interval * 1.5, 2.0)
        return None

    def _sync_after_fill(self, order: Order):
        """Update local state after a confirmed fill."""
        try:
            self._sync_balances()
            self._sync_positions()
        except Exception as e:
            logger.warning("Post-fill sync failed: %s", e)
            # Fall back to local accounting (same as PaperBroker)
            self._local_update_position(order)

        self.state.total_trades += 1

    def _local_update_position(self, order: Order):
        """Local position update as fallback when Tradier sync fails."""
        ticker = order.ticker
        qty = order.quantity
        price = order.fill_price
        pos = self.state.positions.get(ticker, Position(ticker=ticker))

        if order.side == OrderSide.BUY:
            cost = qty * price
            total_qty = pos.quantity + qty
            if total_qty > 0:
                pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty
            self.state.cash -= cost

        elif order.side == OrderSide.SELL:
            sell_qty = min(qty, pos.quantity)
            proceeds = sell_qty * price
            pnl = (price - pos.avg_cost) * sell_qty
            pos.realized_pnl += pnl
            pos.quantity -= sell_qty
            self.state.cash += proceeds
            self.state.total_pnl += pnl
            self._daily_pnl_today += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        elif order.side == OrderSide.SHORT:
            pos.quantity -= qty
            pos.avg_cost = price
            self.state.cash += qty * price

        elif order.side == OrderSide.COVER:
            cover_qty = min(qty, abs(pos.quantity))
            cost = cover_qty * price
            pnl = (pos.avg_cost - price) * cover_qty
            pos.quantity += cover_qty
            pos.realized_pnl += pnl
            self.state.cash -= cost
            self.state.total_pnl += pnl
            self._daily_pnl_today += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        pos.current_price = price
        pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

        if pos.quantity != 0:
            self.state.positions[ticker] = pos
        elif ticker in self.state.positions:
            del self.state.positions[ticker]

    # --- Portfolio state (PaperBroker-compatible interface) -------------------

    def refresh_prices(self):
        """Update all position prices from Tradier real-time quotes."""
        tickers = list(self.state.positions.keys())
        if not tickers:
            return
        try:
            quotes = self.client.get_quotes(tickers)
            quote_map = {q["symbol"]: q for q in quotes if "symbol" in q}
            for ticker in tickers:
                if ticker in quote_map and ticker in self.state.positions:
                    q = quote_map[ticker]
                    price = float(q.get("last", 0) or q.get("close", 0) or 0)
                    if price > 0:
                        pos = self.state.positions[ticker]
                        pos.current_price = price
                        pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
        except Exception as e:
            logger.warning("Tradier quote refresh failed: %s", e)

    def compute_nav(self) -> float:
        """Compute current NAV (syncs from Tradier when possible)."""
        try:
            self._sync_balances()
            return self.state.nav
        except Exception:
            # Fallback: compute locally
            self.refresh_prices()
            positions_value = sum(
                p.quantity * p.current_price
                for p in self.state.positions.values()
            )
            self.state.nav = self.state.cash + positions_value
            return self.state.nav

    def compute_exposures(self) -> dict:
        nav = self.compute_nav()
        if nav <= 0:
            return {"gross": 0, "net": 0, "long": 0, "short": 0}
        long_val = sum(
            p.market_value for p in self.state.positions.values()
            if p.quantity > 0
        )
        short_val = sum(
            abs(p.market_value) for p in self.state.positions.values()
            if p.quantity < 0
        )
        self.state.gross_exposure = (long_val + short_val) / nav
        self.state.net_exposure = (long_val - short_val) / nav
        return {
            "gross": self.state.gross_exposure,
            "net": self.state.net_exposure,
            "long": long_val / nav,
            "short": short_val / nav,
        }

    def get_position(self, ticker: str) -> Optional[Position]:
        return self.state.positions.get(ticker)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self.state.positions)

    def get_portfolio_summary(self) -> dict:
        nav = self.compute_nav()
        exp = self.compute_exposures()
        win_rate = (
            self.state.win_count / (self.state.win_count + self.state.loss_count)
            if (self.state.win_count + self.state.loss_count) > 0 else 0
        )
        return {
            "nav": nav,
            "cash": self.state.cash,
            "total_pnl": self.state.total_pnl,
            "total_trades": self.state.total_trades,
            "win_rate": win_rate,
            "positions": len(self.state.positions),
            "gross_exposure": exp["gross"],
            "net_exposure": exp["net"],
            "broker": "tradier",
            "environment": self.client.environment,
        }

    # --- Price fetching (Tradier real-time quotes) ---------------------------

    def _get_current_price(self, ticker: str) -> float:
        """Get latest price via Tradier quotes API with short cache."""
        now = time.time()
        if ticker in self._quote_cache:
            cached_time, cached_price = self._quote_cache[ticker]
            if now - cached_time < self._quote_cache_ttl:
                return cached_price

        try:
            quote = self.client.get_quote(ticker)
            price = float(quote.get("last", 0) or quote.get("close", 0) or 0)
            if price > 0:
                self._quote_cache[ticker] = (now, price)
                return price
        except Exception as e:
            logger.debug("Tradier quote for %s failed: %s", ticker, e)

        # Fallback to OpenBB if Tradier quote unavailable
        try:
            from ..data.yahoo_data import get_adj_close
            prices = get_adj_close(ticker, start=(
                pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            )
            if not prices.empty:
                price = float(
                    prices.iloc[-1].iloc[0]
                    if isinstance(prices, pd.DataFrame)
                    else prices.iloc[-1]
                )
                if price > 0:
                    self._quote_cache[ticker] = (now, price)
                    return price
        except Exception:
            pass

        return 0.0

    # --- Daily P&L -----------------------------------------------------------

    def get_daily_pnl(self) -> pd.DataFrame:
        return self._perf_tracker.get_daily_pnl_series()

    def get_drawdown(self) -> dict:
        return self._perf_tracker.get_drawdown()

    def get_performance_metrics(self) -> dict:
        metrics = self._perf_tracker.get_all_metrics()
        total = self.state.win_count + self.state.loss_count
        metrics["overall_win_rate"] = (
            self.state.win_count / total if total > 0 else 0.0
        )
        metrics["total_pnl"] = self.state.total_pnl
        metrics["nav"] = self.state.nav
        metrics["cash"] = self.state.cash
        metrics["broker"] = "tradier"
        metrics["environment"] = self.client.environment
        return metrics

    # --- EOD reconciliation --------------------------------------------------

    def reconcile(self) -> dict:
        """End-of-day reconciliation via Tradier API."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Full sync from Tradier
        try:
            self._sync_balances()
            self._sync_positions()
        except Exception as e:
            logger.warning("Tradier EOD sync failed: %s", e)

        nav = self.state.nav
        exposures = self.compute_exposures()
        self._perf_tracker.record_nav(nav, date_str=today)

        # Fetch realized P&L from Tradier
        try:
            gainloss = self.client.get_gainloss()
            today_realized = sum(
                float(gl.get("gain_loss", 0))
                for gl in gainloss
                if gl.get("close_date", "").startswith(today)
            )
        except Exception:
            today_realized = self._daily_pnl_today

        summary = {
            "date": today,
            "nav": nav,
            "cash": self.state.cash,
            "positions_count": len(self.state.positions),
            "total_pnl": self.state.total_pnl,
            "daily_pnl": today_realized,
            "gross_exposure": exposures["gross"],
            "net_exposure": exposures["net"],
            "broker": "tradier",
            "environment": self.client.environment,
            "total_trades_today": sum(
                1 for o in self._orders
                if o.status == OrderStatus.FILLED
                and o.fill_timestamp.startswith(today)
            ),
        }

        self._eod_nav_history.append((today, nav))

        recon_file = self.log_dir / f"reconcile_{today}.json"
        try:
            with open(recon_file, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

        self._daily_pnl_today = 0.0
        self._last_eod_date = today
        self._target_manager.reset_day(nav)
        self.emit_dashboard_state(pipeline_state={"event": "EOD_RECONCILE"})

        return summary

    # --- Position export -----------------------------------------------------

    def export_positions_csv(self, filepath: Optional[str] = None) -> str:
        """Export current positions to CSV (same as PaperBroker)."""
        import csv
        import io
        self.refresh_prices()
        headers = [
            "ticker", "quantity", "avg_cost", "current_price",
            "market_value", "unrealized_pnl", "realized_pnl", "sector",
        ]
        rows = []
        for ticker, pos in sorted(self.state.positions.items()):
            rows.append([
                ticker, pos.quantity,
                round(pos.avg_cost, 4), round(pos.current_price, 4),
                round(pos.market_value, 2), round(pos.unrealized_pnl, 2),
                round(pos.realized_pnl, 2), pos.sector,
            ])
        if filepath:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            return str(path)
        else:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(headers)
            writer.writerows(rows)
            return buf.getvalue()

    # --- Logging -------------------------------------------------------------

    def _log_trade(self, order: Order):
        """Log a trade with extended metadata."""
        entry = order.to_dict()
        entry["nav_at_fill"] = self.state.nav
        entry["cash_after"] = self.state.cash
        entry["positions_count"] = len(self.state.positions)
        entry["daily_pnl"] = self._daily_pnl_today
        entry["total_pnl"] = self.state.total_pnl
        entry["total_trades"] = self.state.total_trades
        entry["broker"] = "tradier"
        entry["environment"] = self.client.environment

        if order.ticker in self.state.positions:
            pos = self.state.positions[order.ticker]
            entry["pos_quantity"] = pos.quantity
            entry["pos_avg_cost"] = pos.avg_cost
            entry["pos_unrealized_pnl"] = pos.unrealized_pnl
            entry["pos_realized_pnl"] = pos.realized_pnl
            entry["pos_sector"] = pos.sector
        else:
            entry["pos_quantity"] = 0
            entry["pos_avg_cost"] = 0.0
            entry["pos_unrealized_pnl"] = 0.0
            entry["pos_realized_pnl"] = 0.0
            entry["pos_sector"] = ""

        # Tradier order ID
        tradier_id = self._order_id_map.get(order.id, "")
        entry["tradier_order_id"] = tradier_id

        realized = 0.0
        if order.side in (OrderSide.SELL, OrderSide.COVER):
            realized = entry.get("pos_realized_pnl", 0.0)
        perf_entry = dict(entry)
        perf_entry["realized_pnl"] = realized
        self._perf_tracker.record_trade(perf_entry)

        self._trade_log.append(entry)
        log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_trade_history(self) -> list[dict]:
        return list(self._trade_log)

    # --- Daily target & risk profile (same as PaperBroker) -------------------

    def get_risk_profile(self) -> str:
        return self._target_manager.profile.value

    def get_daily_target_state(self) -> dict:
        nav = self.compute_nav()
        self._target_manager.update(nav)
        state = self._target_manager.get_state()
        state["current_nav"] = nav
        if self._target_manager._initial_nav_today > 0:
            state["daily_return_pct"] = (
                (nav - self._target_manager._initial_nav_today)
                / self._target_manager._initial_nav_today
            )
        else:
            state["daily_return_pct"] = 0.0
        return state

    def reset_daily_target(self):
        nav = self.compute_nav()
        self._target_manager.reset_day(nav)

    def get_leverage_multiplier(self) -> float:
        return self._target_manager.get_leverage_multiplier()

    # --- Live dashboard (same as PaperBroker) --------------------------------

    def emit_dashboard_state(self, pipeline_state: Optional[dict] = None):
        self._dashboard.emit(
            broker_state=self.get_portfolio_summary(),
            target_state=self.get_daily_target_state(),
            pipeline_state=pipeline_state,
        )

    def get_dashboard_snapshot(self) -> dict:
        return self._dashboard.get_latest()

    def get_dashboard_history(self, n: int = 100) -> list[dict]:
        return self._dashboard.get_history(n)

    def register_dashboard_callback(self, callback):
        self._dashboard.register_callback(callback)

    # --- Tradier-specific methods --------------------------------------------

    def get_tradier_orders(self) -> list[dict]:
        """Fetch all orders directly from Tradier API."""
        return self.client.get_orders()

    def cancel_tradier_order(self, tradier_order_id: str) -> dict:
        """Cancel an order on Tradier by its Tradier order ID."""
        return self.client.cancel_order(tradier_order_id)

    def get_tradier_gainloss(self) -> list[dict]:
        """Fetch realized gain/loss report from Tradier."""
        return self.client.get_gainloss()

    def preview_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> dict:
        """Preview an order without executing (Tradier preview mode)."""
        tradier_side = _SIDE_TO_TRADIER[side]
        order_type = "limit" if limit_price else "market"
        return self.client.place_equity_order(
            symbol=ticker,
            side=tradier_side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            preview=True,
        )
