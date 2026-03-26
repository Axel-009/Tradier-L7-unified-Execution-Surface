"""SocialPredictionEngine — MiroFish integration for social sentiment signals.

Layer 2 signal source that bridges MiroFish agent-based social simulations
into the Metadron Capital signal pipeline.

MiroFish runs multi-agent social media simulations (Twitter/Reddit) where
LLM-powered agents interact, post, like, repost, and comment on topics.
This engine reads simulation output (actions.jsonl) and converts agent
behavior patterns into actionable trading signals:

Signal Types Generated:
    SOCIAL_BULLISH  — Consensus bullish sentiment across agents
    SOCIAL_BEARISH  — Consensus bearish sentiment across agents
    SOCIAL_MOMENTUM — Viral spread velocity indicates trend acceleration
    SOCIAL_REVERSAL — Stance flips / narrative shift detected

Output: SocialSnapshot consumed by MLVoteEnsemble (Tier-6) and AlphaOptimizer.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """Single agent action from MiroFish simulation."""
    round_num: int = 0
    timestamp: str = ""
    platform: str = ""
    agent_id: int = 0
    agent_name: str = ""
    action_type: str = ""
    action_args: dict = field(default_factory=dict)
    result: str = ""
    success: bool = True


@dataclass
class TopicSentiment:
    """Aggregated sentiment for a single topic/keyword."""
    topic: str = ""
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    total_engagement: int = 0
    viral_velocity: float = 0.0      # reposts per hour
    echo_chamber: float = 0.0        # stance homogeneity [0,1]
    sentiment_score: float = 0.0     # [-1, +1]
    confidence: float = 0.0          # [0, 1]
    stance_flip_count: int = 0       # narrative reversals detected


@dataclass
class AgentProfile:
    """Summary of an agent's behavior during simulation."""
    agent_id: int = 0
    agent_name: str = ""
    total_actions: int = 0
    sentiment_bias: float = 0.0      # [-1, +1]
    influence_score: float = 0.0     # [0, 1]
    stance: str = "neutral"
    active_rounds: int = 0
    post_count: int = 0
    like_count: int = 0
    repost_count: int = 0
    comment_count: int = 0


@dataclass
class SocialSnapshot:
    """Complete social prediction output consumed by pipeline.

    This is the primary output of SocialPredictionEngine, analogous to
    MacroSnapshot from MacroEngine.
    """
    timestamp: str = ""
    simulation_id: str = ""
    platform: str = ""
    total_actions: int = 0
    total_agents: int = 0
    simulation_hours: int = 0

    # Aggregate sentiment
    overall_sentiment: float = 0.0        # [-1, +1]
    sentiment_confidence: float = 0.0     # [0, 1]
    sentiment_trend: float = 0.0          # rate of change
    narrative_dominance: float = 0.0      # how one-sided [0, 1]

    # Momentum signals
    viral_velocity: float = 0.0           # avg reposts/hr
    engagement_momentum: float = 0.0      # trend in engagement
    influence_concentration: float = 0.0  # HHI of agent influence

    # Topic-level detail
    topic_sentiments: dict = field(default_factory=dict)  # topic -> TopicSentiment

    # Agent-level summary
    agent_count_bullish: int = 0
    agent_count_bearish: int = 0
    agent_count_neutral: int = 0

    # Derived trading signals
    social_signal: str = "HOLD"           # SOCIAL_BULLISH/BEARISH/MOMENTUM/REVERSAL
    signal_strength: float = 0.0          # [0, 1]
    vote_score: int = 0                   # [-1, 0, +1] for ML ensemble

    # Sector/ticker mapping (if detected from topics)
    ticker_signals: dict = field(default_factory=dict)  # ticker -> sentiment float


# ---------------------------------------------------------------------------
# Keyword → Ticker Mapping
# ---------------------------------------------------------------------------
# Maps social media topics/keywords to trading tickers
TOPIC_TICKER_MAP = {
    # Tech
    "apple": "AAPL", "iphone": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "windows": "MSFT", "azure": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "amazon": "AMZN", "aws": "AMZN", "amzn": "AMZN",
    "meta": "META", "facebook": "META", "instagram": "META",
    "nvidia": "NVDA", "nvda": "NVDA", "gpu": "NVDA", "cuda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA", "elon": "TSLA",
    "netflix": "NFLX", "nflx": "NFLX",
    # Finance
    "jpmorgan": "JPM", "jpm": "JPM", "chase": "JPM",
    "goldman": "GS", "gs": "GS",
    "bank of america": "BAC", "bac": "BAC",
    # Sectors
    "tech": "XLK", "technology": "XLK",
    "financials": "XLF", "banking": "XLF", "banks": "XLF",
    "energy": "XLE", "oil": "XLE", "crude": "XLE",
    "healthcare": "XLV", "pharma": "XLV", "biotech": "XBI",
    "consumer": "XLY", "retail": "XRT",
    "crypto": "BITO", "bitcoin": "BITO", "btc": "BITO",
    # Market
    "sp500": "SPY", "s&p": "SPY", "market": "SPY",
    "nasdaq": "QQQ", "qqq": "QQQ",
    "recession": "TLT", "bonds": "TLT", "fed": "TLT",
    "inflation": "TIP", "cpi": "TIP",
    "gold": "GLD", "safe haven": "GLD",
    "vix": "VIX", "volatility": "VIX", "fear": "VIX",
}

# Bullish/bearish keyword indicators for content analysis
BULLISH_KEYWORDS = {
    "buy", "bullish", "moon", "long", "upgrade", "growth", "beat",
    "outperform", "rally", "surge", "breakout", "up", "gain", "profit",
    "strong", "positive", "boom", "rocket", "soar", "amazing",
}
BEARISH_KEYWORDS = {
    "sell", "bearish", "crash", "short", "downgrade", "decline", "miss",
    "underperform", "dump", "tank", "breakdown", "down", "loss", "risk",
    "weak", "negative", "bust", "plunge", "terrible", "overvalued",
}


# ---------------------------------------------------------------------------
# SocialPredictionEngine
# ---------------------------------------------------------------------------
class SocialPredictionEngine:
    """Bridge between MiroFish social simulations and Metadron signal pipeline.

    Reads simulation output from MiroFish (actions.jsonl files) and produces
    SocialSnapshot objects consumed by the ML Vote Ensemble and alpha pipeline.

    Architecture:
        MiroFish Backend (Flask, port 5001)
            └── simulations/{sim_id}/{platform}/actions.jsonl
                    ↓ (read by this engine)
        SocialPredictionEngine
            ├── _parse_actions()      → List[AgentAction]
            ├── _build_agent_profiles() → Dict[agent_id, AgentProfile]
            ├── _analyze_sentiment()  → per-action sentiment scoring
            ├── _compute_topics()     → TopicSentiment per keyword
            ├── _detect_narratives()  → stance flips, echo chambers
            ├── _map_to_tickers()     → topic → ticker signal mapping
            └── analyze()             → SocialSnapshot
    """

    def __init__(
        self,
        simulation_dir: Optional[str] = None,
        lookback_rounds: int = 0,
        sentiment_threshold: float = 0.3,
        viral_threshold: float = 5.0,
    ):
        # Default to MiroFish uploads directory
        if simulation_dir is None:
            base = Path(__file__).parent.parent.parent / "mirofish" / "backend" / "uploads" / "simulations"
            self.simulation_dir = str(base)
        else:
            self.simulation_dir = simulation_dir

        self.lookback_rounds = lookback_rounds  # 0 = all rounds
        self.sentiment_threshold = sentiment_threshold
        self.viral_threshold = viral_threshold

        self._last_snapshot: Optional[SocialSnapshot] = None
        self._snapshot_history: list[SocialSnapshot] = []

    def analyze(self, simulation_id: Optional[str] = None, platform: str = "twitter") -> SocialSnapshot:
        """Run full social prediction analysis.

        If simulation_id is None, finds the latest simulation.
        Returns SocialSnapshot with all derived signals.
        """
        snap = SocialSnapshot(
            timestamp=datetime.now().isoformat(),
            platform=platform,
        )

        # Find simulation directory
        sim_dir = self._find_simulation_dir(simulation_id)
        if sim_dir is None:
            logger.warning("No MiroFish simulation data found — returning neutral snapshot")
            snap.social_signal = "HOLD"
            snap.vote_score = 0
            self._last_snapshot = snap
            return snap

        snap.simulation_id = os.path.basename(sim_dir)

        # Parse actions
        actions = self._parse_actions(sim_dir, platform)
        if not actions:
            logger.warning(f"No actions found for simulation {snap.simulation_id}/{platform}")
            self._last_snapshot = snap
            return snap

        snap.total_actions = len(actions)
        snap.simulation_hours = max(1, max(a.round_num for a in actions))

        # Build agent profiles
        profiles = self._build_agent_profiles(actions)
        snap.total_agents = len(profiles)

        # Classify agents
        for p in profiles.values():
            if p.sentiment_bias > self.sentiment_threshold:
                snap.agent_count_bullish += 1
            elif p.sentiment_bias < -self.sentiment_threshold:
                snap.agent_count_bearish += 1
            else:
                snap.agent_count_neutral += 1

        # Compute overall sentiment
        if profiles:
            sentiments = [p.sentiment_bias for p in profiles.values()]
            influences = [p.influence_score for p in profiles.values()]
            total_influence = sum(influences) or 1.0

            # Influence-weighted sentiment
            snap.overall_sentiment = sum(
                s * w for s, w in zip(sentiments, influences)
            ) / total_influence

            snap.sentiment_confidence = 1.0 - np.std(sentiments) if len(sentiments) > 1 else 0.5
            snap.sentiment_confidence = max(0.0, min(1.0, snap.sentiment_confidence))

            # Narrative dominance (how one-sided)
            if snap.total_agents > 0:
                majority = max(snap.agent_count_bullish, snap.agent_count_bearish)
                snap.narrative_dominance = majority / snap.total_agents

            # Influence concentration (HHI)
            if total_influence > 0:
                shares = [w / total_influence for w in influences]
                snap.influence_concentration = sum(s ** 2 for s in shares)

        # Compute topic sentiments
        snap.topic_sentiments = self._compute_topics(actions)

        # Map topics to tickers
        snap.ticker_signals = self._map_to_tickers(snap.topic_sentiments)

        # Compute momentum signals
        snap.viral_velocity = self._compute_viral_velocity(actions)
        snap.engagement_momentum = self._compute_engagement_momentum(actions)

        # Detect sentiment trend (first half vs second half)
        snap.sentiment_trend = self._compute_sentiment_trend(actions)

        # Derive final trading signal
        self._derive_signal(snap)

        # Store
        self._last_snapshot = snap
        self._snapshot_history.append(snap)
        if len(self._snapshot_history) > 100:
            self._snapshot_history = self._snapshot_history[-50:]

        return snap

    def get_last_snapshot(self) -> Optional[SocialSnapshot]:
        return self._last_snapshot

    def get_ticker_signal(self, ticker: str) -> float:
        """Get social sentiment for a specific ticker. Returns [-1, +1]."""
        if self._last_snapshot is None:
            return 0.0
        return self._last_snapshot.ticker_signals.get(ticker, 0.0)

    def get_vote(self) -> int:
        """Get aggregate vote for ML ensemble. Returns -1, 0, or +1."""
        if self._last_snapshot is None:
            return 0
        return self._last_snapshot.vote_score

    # -------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------

    def _find_simulation_dir(self, simulation_id: Optional[str]) -> Optional[str]:
        """Find simulation directory. If None, returns latest by mtime."""
        base = Path(self.simulation_dir)
        if not base.exists():
            return None

        if simulation_id:
            target = base / simulation_id
            return str(target) if target.exists() else None

        # Find latest simulation directory
        sim_dirs = [d for d in base.iterdir() if d.is_dir()]
        if not sim_dirs:
            return None

        # Sort by modification time (newest first)
        sim_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return str(sim_dirs[0])

    def _parse_actions(self, sim_dir: str, platform: str) -> list[AgentAction]:
        """Parse actions.jsonl from a simulation run."""
        actions_file = Path(sim_dir) / platform / "actions.jsonl"

        # Also try root-level actions file
        if not actions_file.exists():
            actions_file = Path(sim_dir) / "actions.jsonl"
        if not actions_file.exists():
            return []

        actions = []
        try:
            with open(actions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        action = AgentAction(
                            round_num=data.get("round_num", 0),
                            timestamp=data.get("timestamp", ""),
                            platform=data.get("platform", platform),
                            agent_id=data.get("agent_id", 0),
                            agent_name=data.get("agent_name", ""),
                            action_type=data.get("action_type", ""),
                            action_args=data.get("action_args", {}),
                            result=data.get("result", ""),
                            success=data.get("success", True),
                        )
                        actions.append(action)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except (IOError, OSError) as e:
            logger.error(f"Failed to read actions file: {e}")

        # Apply lookback filter
        if self.lookback_rounds > 0 and actions:
            max_round = max(a.round_num for a in actions)
            cutoff = max_round - self.lookback_rounds
            actions = [a for a in actions if a.round_num >= cutoff]

        return actions

    def _build_agent_profiles(self, actions: list[AgentAction]) -> dict[int, AgentProfile]:
        """Build per-agent behavioral profiles from action stream."""
        profiles: dict[int, AgentProfile] = {}

        for a in actions:
            if a.agent_id not in profiles:
                profiles[a.agent_id] = AgentProfile(
                    agent_id=a.agent_id,
                    agent_name=a.agent_name,
                )

            p = profiles[a.agent_id]
            p.total_actions += 1

            if a.action_type == "CREATE_POST":
                p.post_count += 1
                # Analyze content sentiment
                content = a.action_args.get("content", "")
                p.sentiment_bias = self._update_running_sentiment(
                    p.sentiment_bias, content, p.post_count,
                )
            elif a.action_type in ("LIKE_POST", "LIKE_COMMENT"):
                p.like_count += 1
            elif a.action_type in ("DISLIKE_POST", "DISLIKE_COMMENT"):
                p.like_count -= 1  # negative engagement
            elif a.action_type in ("REPOST", "QUOTE_POST"):
                p.repost_count += 1
            elif a.action_type == "CREATE_COMMENT":
                p.comment_count += 1
                content = a.action_args.get("content", "")
                p.sentiment_bias = self._update_running_sentiment(
                    p.sentiment_bias, content, p.comment_count,
                )

        # Compute influence scores
        if profiles:
            max_actions = max(p.total_actions for p in profiles.values()) or 1
            max_reposts = max(p.repost_count for p in profiles.values()) or 1
            for p in profiles.values():
                action_share = p.total_actions / max_actions
                repost_share = p.repost_count / max_reposts
                p.influence_score = 0.5 * action_share + 0.5 * repost_share

                # Classify stance
                if p.sentiment_bias > self.sentiment_threshold:
                    p.stance = "bullish"
                elif p.sentiment_bias < -self.sentiment_threshold:
                    p.stance = "bearish"
                else:
                    p.stance = "neutral"

        return profiles

    def _update_running_sentiment(self, current: float, content: str, n: int) -> float:
        """Update running sentiment score from content analysis."""
        if not content:
            return current

        words = set(content.lower().split())
        bull_hits = len(words & BULLISH_KEYWORDS)
        bear_hits = len(words & BEARISH_KEYWORDS)
        total_hits = bull_hits + bear_hits

        if total_hits == 0:
            return current

        new_sentiment = (bull_hits - bear_hits) / total_hits  # [-1, +1]

        # Exponential moving average
        alpha = 2.0 / (n + 1) if n > 0 else 1.0
        return current * (1 - alpha) + new_sentiment * alpha

    def _compute_topics(self, actions: list[AgentAction]) -> dict[str, TopicSentiment]:
        """Extract topic-level sentiment from actions."""
        topic_data: dict[str, TopicSentiment] = {}

        for a in actions:
            content = a.action_args.get("content", "")
            if not content:
                continue

            words = set(content.lower().split())

            # Match against known topic keywords
            matched_topics = set()
            for keyword, ticker in TOPIC_TICKER_MAP.items():
                if keyword in content.lower():
                    matched_topics.add(ticker)

            # Also detect generic topics from hot_topic tags
            tags = a.action_args.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    matched_topics.add(tag.lower())

            for topic in matched_topics:
                if topic not in topic_data:
                    topic_data[topic] = TopicSentiment(topic=topic)

                ts = topic_data[topic]
                ts.total_engagement += 1

                # Classify action sentiment
                bull_hits = len(words & BULLISH_KEYWORDS)
                bear_hits = len(words & BEARISH_KEYWORDS)

                if a.action_type in ("LIKE_POST", "LIKE_COMMENT"):
                    ts.bullish_count += 1
                elif a.action_type in ("DISLIKE_POST", "DISLIKE_COMMENT"):
                    ts.bearish_count += 1
                elif bull_hits > bear_hits:
                    ts.bullish_count += 1
                elif bear_hits > bull_hits:
                    ts.bearish_count += 1
                else:
                    ts.neutral_count += 1

                if a.action_type in ("REPOST", "QUOTE_POST"):
                    ts.viral_velocity += 1

        # Normalize scores
        for ts in topic_data.values():
            total = ts.bullish_count + ts.bearish_count + ts.neutral_count
            if total > 0:
                ts.sentiment_score = (ts.bullish_count - ts.bearish_count) / total
                ts.confidence = 1.0 - (ts.neutral_count / total)
            # Echo chamber = how one-sided
            majority = max(ts.bullish_count, ts.bearish_count)
            ts.echo_chamber = majority / total if total > 0 else 0.0

        return topic_data

    def _map_to_tickers(self, topics: dict[str, TopicSentiment]) -> dict[str, float]:
        """Map topic sentiments to ticker-level signals."""
        ticker_signals: dict[str, list[float]] = {}

        for topic, ts in topics.items():
            # If topic is already a ticker symbol
            ticker = topic.upper()
            if ticker not in ticker_signals:
                ticker_signals[ticker] = []
            ticker_signals[ticker].append(ts.sentiment_score * ts.confidence)

        # Average multiple signals per ticker
        return {
            ticker: float(np.mean(scores)) if scores else 0.0
            for ticker, scores in ticker_signals.items()
        }

    def _compute_viral_velocity(self, actions: list[AgentAction]) -> float:
        """Compute average reposts per simulated hour."""
        repost_types = {"REPOST", "QUOTE_POST"}
        reposts = [a for a in actions if a.action_type in repost_types]
        if not actions:
            return 0.0
        max_round = max(a.round_num for a in actions)
        hours = max(1, max_round)
        return len(reposts) / hours

    def _compute_engagement_momentum(self, actions: list[AgentAction]) -> float:
        """Compute engagement momentum (second half vs first half)."""
        if len(actions) < 10:
            return 0.0
        mid = len(actions) // 2
        first_half = actions[:mid]
        second_half = actions[mid:]

        engagement_first = sum(
            1 for a in first_half
            if a.action_type in {"LIKE_POST", "REPOST", "CREATE_COMMENT", "QUOTE_POST"}
        ) / max(1, len(first_half))

        engagement_second = sum(
            1 for a in second_half
            if a.action_type in {"LIKE_POST", "REPOST", "CREATE_COMMENT", "QUOTE_POST"}
        ) / max(1, len(second_half))

        if engagement_first == 0:
            return 0.0
        return (engagement_second - engagement_first) / engagement_first

    def _compute_sentiment_trend(self, actions: list[AgentAction]) -> float:
        """Compute sentiment trend: positive = getting more bullish."""
        content_actions = [a for a in actions if a.action_args.get("content")]
        if len(content_actions) < 10:
            return 0.0

        mid = len(content_actions) // 2
        first_half = content_actions[:mid]
        second_half = content_actions[mid:]

        def avg_sentiment(action_list):
            scores = []
            for a in action_list:
                content = a.action_args.get("content", "")
                words = set(content.lower().split())
                bull = len(words & BULLISH_KEYWORDS)
                bear = len(words & BEARISH_KEYWORDS)
                total = bull + bear
                if total > 0:
                    scores.append((bull - bear) / total)
            return np.mean(scores) if scores else 0.0

        return avg_sentiment(second_half) - avg_sentiment(first_half)

    def _derive_signal(self, snap: SocialSnapshot) -> None:
        """Derive final trading signal from social snapshot."""
        sentiment = snap.overall_sentiment
        confidence = snap.sentiment_confidence
        trend = snap.sentiment_trend
        dominance = snap.narrative_dominance

        # Default
        snap.social_signal = "HOLD"
        snap.signal_strength = 0.0
        snap.vote_score = 0

        # Strong consensus bullish
        if sentiment > self.sentiment_threshold and confidence > 0.5:
            if trend > 0.1:
                snap.social_signal = "SOCIAL_MOMENTUM"
                snap.signal_strength = min(1.0, abs(sentiment) * confidence * (1 + trend))
            else:
                snap.social_signal = "SOCIAL_BULLISH"
                snap.signal_strength = abs(sentiment) * confidence
            snap.vote_score = 1

        # Strong consensus bearish
        elif sentiment < -self.sentiment_threshold and confidence > 0.5:
            if trend < -0.1:
                snap.social_signal = "SOCIAL_MOMENTUM"
                snap.signal_strength = min(1.0, abs(sentiment) * confidence * (1 + abs(trend)))
            else:
                snap.social_signal = "SOCIAL_BEARISH"
                snap.signal_strength = abs(sentiment) * confidence
            snap.vote_score = -1

        # Narrative reversal detection
        elif abs(trend) > 0.3 and dominance > 0.6:
            snap.social_signal = "SOCIAL_REVERSAL"
            snap.signal_strength = abs(trend) * dominance
            snap.vote_score = 1 if trend > 0 else -1

        # Viral but ambiguous — elevated activity signal
        elif snap.viral_velocity > self.viral_threshold:
            snap.social_signal = "SOCIAL_MOMENTUM"
            snap.signal_strength = min(1.0, snap.viral_velocity / (self.viral_threshold * 3))
            snap.vote_score = 1 if sentiment >= 0 else -1

    def as_dict(self) -> dict:
        """Return last snapshot as dict for pipeline stage output."""
        if self._last_snapshot is None:
            return {"status": "no_simulation_data"}

        s = self._last_snapshot
        return {
            "simulation_id": s.simulation_id,
            "platform": s.platform,
            "total_actions": s.total_actions,
            "total_agents": s.total_agents,
            "overall_sentiment": round(s.overall_sentiment, 4),
            "sentiment_confidence": round(s.sentiment_confidence, 4),
            "sentiment_trend": round(s.sentiment_trend, 4),
            "narrative_dominance": round(s.narrative_dominance, 4),
            "viral_velocity": round(s.viral_velocity, 4),
            "engagement_momentum": round(s.engagement_momentum, 4),
            "social_signal": s.social_signal,
            "signal_strength": round(s.signal_strength, 4),
            "vote_score": s.vote_score,
            "agents_bullish": s.agent_count_bullish,
            "agents_bearish": s.agent_count_bearish,
            "agents_neutral": s.agent_count_neutral,
            "ticker_signals": {k: round(v, 4) for k, v in s.ticker_signals.items()},
            "top_topics": sorted(
                [
                    {"topic": ts.topic, "sentiment": round(ts.sentiment_score, 4), "engagement": ts.total_engagement}
                    for ts in s.topic_sentiments.values()
                ],
                key=lambda x: x["engagement"],
                reverse=True,
            )[:10],
        }
