"""Unified data provider for the entire platform.

Delegates to OpenBB as the sole data source.
All existing imports from this module continue to work unchanged.

OpenBB provides access to 34+ data providers (FRED, SEC, Polygon, FMP,
Intrinio, CBOE, ECB, OECD, etc.) through a single unified API.
"""

# ---------------------------------------------------------------------------
# Re-export everything from openbb_data — backward-compatible interface.
# Every module that does `from ..data.yahoo_data import get_prices` etc.
# continues to work without any import changes.
# ---------------------------------------------------------------------------

from .openbb_data import (
    # Core price/return functions (existing interface)
    get_prices,
    get_adj_close,
    get_returns,
    get_market_stats,
    # Fundamental data (existing interface)
    get_fundamentals,
    get_bulk_fundamentals,
    # Macro data (existing interface)
    get_macro_data,
    get_sector_performance,
    MACRO_PROXIES,
    # ── NEW: OpenBB-powered data sources ──
    # FRED economic data (replaces ETF proxies with real data)
    get_fred_series,
    get_treasury_rates,
    get_fed_balance_sheet,
    get_credit_spreads,
    get_monetary_data,
    get_sofr_rate,
    get_effr_rate,
    get_cpi,
    get_unemployment,
    # Enriched macro bundle
    get_macro_data_enriched,
    # SEC filings
    get_company_filings,
    get_insider_trading,
    # News
    get_company_news,
    get_world_news,
    # Options
    get_options_chains,
    # ETF data
    get_etf_holdings,
    # Economic calendar
    get_economic_calendar,
    # Equity search
    search_equities,
    # Diagnostics
    get_data_source_status,
    # Constants
    FRED_SERIES,
)
