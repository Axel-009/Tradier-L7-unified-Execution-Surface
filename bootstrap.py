#!/usr/bin/env python3
"""
Metadron Capital — Session Bootstrap Script
Run this at the start of any new session to verify platform state.

Usage:
    python3 bootstrap.py
    python3 bootstrap.py --full    # Run tests too
"""

import sys
import os
import subprocess
from pathlib import Path

PLATFORM_ROOT = Path(__file__).parent
REPOS_DIR = PLATFORM_ROOT / "repos"
INTEL_DIR = PLATFORM_ROOT / "intelligence_platform"

LAYER_MAP = {
    "layer1_data": {
        "Financial-Data": "OpenBB market data pipeline",
        "open-bb": "OpenBB investment research terminal",
        "hedgefund-tracker": "SEC 13F institutional flow tracker",
        "FRB": "Federal Reserve FRED API data",
        "EquityLinkedGICPooling": "GIC pooling methodology",
        "Quant-Developers-Resources": "Quantitative finance reference library",
    },
    "layer2_signals": {
        "Mav-Analysis": "MaverickMCP technical analysis engine",
        "quant-trading": "Quantitative strategy library",
        "stock-chain": "Time-series chain analysis",
        "CTA-code": "CTA/trend-following signals",
        "TradeTheEvent": "Event-driven ML (BERT news classification)",
        "wondertrader": "HFT quantitative trading (C++/Python)",
        "worldmonitor": "Global real-time event monitoring (event + macro engine feed)",
    },
    "layer3_ml": {
        "QLIB": "Microsoft Qlib quantitative ML framework",
        "Stock-techincal-prediction-model": "Technical price prediction models",
        "Stock-prediction": "DL price prediction models",
        "ML-Macro-Market": "Macro-to-market regime classification",
        "AI-Newton": "Physics-inspired financial models",
        "markov-model": "Hidden Markov Model regime detection (hmmlearn)",
    },
    "layer4_portfolio": {
        "ai-hedgefund": "Multi-agent AI hedge fund",
        "financial-distressed-repo": "Company default prediction",
        "sophisticated-distress-analysis": "Advanced distress analytics",
        "FinancialDistressPrediction": "GBM bankruptcy prediction reference",
    },
    "layer5_infra": {
        "Kserve": "Kubernetes ML model serving",
        "nividia-repo": "NVIDIA GPU-optimized DL",
        "Air-LLM": "Memory-efficient LLM inference",
        "exchange-core": "Ultra-low latency order matching (Java)",
    },
    "layer6_agents": {
        "Ruflo-agents": "claude-flow multi-agent orchestration",
        "MiroFish": "Agent-based social prediction engine",
    },
}


def check_repo(layer: str, repo: str) -> tuple[bool, int]:
    """Check if a repo exists in repos/, repos/{layer}/, or intelligence_platform/ and count files."""
    repo_path = REPOS_DIR / layer / repo
    repo_flat_path = REPOS_DIR / repo
    intel_path = INTEL_DIR / repo

    # Check all locations
    path = None
    if repo_path.exists():
        path = repo_path
    elif repo_flat_path.exists():
        path = repo_flat_path
    elif intel_path.exists():
        path = intel_path

    if path is None:
        return False, 0
    file_count = sum(1 for _ in path.rglob("*") if _.is_file() and "__pycache__" not in str(_))
    return True, file_count


def check_intelligence_platform() -> tuple[int, int]:
    """Check intelligence_platform completeness."""
    if not INTEL_DIR.exists():
        return 0, 0
    repos = [d for d in INTEL_DIR.iterdir() if d.is_dir() and d.name != "plugins" and d.name != "__pycache__"]
    total_files = sum(
        sum(1 for _ in r.rglob("*") if _.is_file() and "__pycache__" not in str(_))
        for r in repos
    )
    return len(repos), total_files


def check_core() -> bool:
    """Verify core platform modules."""
    try:
        sys.path.insert(0, str(PLATFORM_ROOT))
        from core.platform import MetadronPlatform
        from core.signals import SignalEngine
        from core.portfolio import PortfolioEngine
        return True
    except ImportError as e:
        print(f"  Core import error: {e}")
        return False


def run_tests() -> int:
    """Run platform tests."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(PLATFORM_ROOT),
        capture_output=True, text=True
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return result.returncode


def main():
    full_mode = "--full" in sys.argv

    print("=" * 70)
    print("METADRON CAPITAL — SESSION BOOTSTRAP")
    print("=" * 70)
    print()

    # Check core
    print("Checking core platform modules...")
    core_ok = check_core()
    print(f"  Core: {'OK' if core_ok else 'FAILED'}")
    print()

    # Check intelligence platform
    intel_repos, intel_files = check_intelligence_platform()
    print(f"Intelligence Platform: {intel_repos} repos, {intel_files:,} files")
    print()

    # Check repos by layer
    total_files = 0
    total_repos = 0
    missing_repos = []
    expected_repos = sum(len(repos) for repos in LAYER_MAP.values())

    for layer, repos in LAYER_MAP.items():
        layer_name = layer.replace("_", " ").upper()
        print(f"━━━ {layer_name} ━━━")
        for repo, description in repos.items():
            exists, count = check_repo(layer, repo)
            total_files += count
            if exists:
                total_repos += 1
                print(f"  ● {repo:<40} {count:>6} files  [{description}]")
            else:
                missing_repos.append(repo)
                print(f"  ✗ {repo:<40} MISSING")
        print()

    # Summary
    print("=" * 70)
    print(f"REPOS: {total_repos}/{expected_repos} present | FILES: {total_files:,} | CORE: {'OK' if core_ok else 'FAILED'}")
    if missing_repos:
        print(f"MISSING: {', '.join(missing_repos)}")
    print("=" * 70)

    if full_mode:
        print("\nRunning platform tests...")
        return run_tests()

    if total_repos == expected_repos and core_ok:
        print("\nPlatform ready. Run with --full to execute tests.")
        return 0
    else:
        print("\nPlatform has issues. Check missing repos or core imports.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
