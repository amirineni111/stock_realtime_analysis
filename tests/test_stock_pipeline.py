"""
Coverage for the learning loop added on top of the rules scorer: direction-aware
structure, the cost/probability gate, the feature contract, the model, and the
storage round trip that turns an armed signal into a training row.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from stocks.features import FEATURE_NAMES, FEATURE_VERSION, build_features, to_vector
from stocks.indicators import calculate_relative_volume
from stocks.model import (
    MIN_TRAIN_SAMPLES, StockModel, brier, build_matrix, calibration_bins, fit,
    gated_performance, roc_auc, top_decile_precision, walk_forward,
)
from stocks.signals import (
    _MAX_COST_RATIO, _PROB_MARGIN, _RR, _sr_proximity, breakeven_win_rate,
    estimate_cost_pct, score_ticker,
)
from stocks.storage import Storage
from stocks.timeutil import minutes_since_open_at, parse_ts, session_progress


# ── Support / resistance ───────────────────────────────────────────────────

class TestSRDirectionAwareness:
    LEVELS = [
        {"price": 99.0, "type": "S", "touches": 3, "strength": 3.0},
        {"price": 101.0, "type": "R", "touches": 3, "strength": 3.0},
    ]

    def test_long_at_support_scores_positive(self):
        # close 99.2, ATR 1.0 → support 0.2 ATR behind, resistance 1.8 ATR ahead
        score, _, at_key, blocked, sup, res = _sr_proximity(99.2, 1.0, self.LEVELS, "LONG")
        assert at_key is True
        assert blocked is False
        assert score > 0
        assert (sup, res) == (99.0, 101.0)

    def test_long_pinned_under_resistance_is_penalised(self):
        """The regression: a long with resistance 0.8 ATR overhead used to score the
        same +25 as a long bouncing off support, because the old scorer never looked
        at which way the trade pointed."""
        score, reason, at_key, blocked, _, _ = _sr_proximity(100.8, 1.0, self.LEVELS, "LONG")
        assert blocked is True
        assert score < 0
        assert "BLOCKED" in reason

    def test_short_mirrors_long(self):
        """Same price, opposite direction: what blocks a long shelters a short."""
        long_score, _, _, long_blocked, _, _ = _sr_proximity(100.8, 1.0, self.LEVELS, "LONG")
        short_score, _, short_at_key, short_blocked, _, _ = _sr_proximity(
            100.8, 1.0, self.LEVELS, "SHORT"
        )
        assert long_blocked and not short_blocked
        assert short_at_key is True
        assert short_score > 0 > long_score

    def test_neutral_direction_scores_nothing_but_still_reports_levels(self):
        score, _, at_key, blocked, sup, res = _sr_proximity(100.0, 1.0, self.LEVELS, "NEUTRAL")
        assert (score, at_key, blocked) == (0.0, False, False)
        assert (sup, res) == (99.0, 101.0)

    def test_score_is_bounded(self):
        for direction in ("LONG", "SHORT"):
            score, *_ = _sr_proximity(100.0, 0.01, self.LEVELS, direction)
            assert -25.0 <= score <= 25.0


# ── Cost arithmetic ────────────────────────────────────────────────────────

class TestCostMath:
    def test_cost_falls_with_liquidity(self):
        mega = estimate_cost_pct(500_000_000)
        mid = estimate_cost_pct(20_000_000)
        small = estimate_cost_pct(5_000_000)
        thin = estimate_cost_pct(500_000)
        assert mega < mid < small < thin

    def test_unknown_liquidity_is_priced_pessimistically(self):
        """An unmeasured cost is not a zero cost."""
        assert estimate_cost_pct(None) == estimate_cost_pct(0)
        assert estimate_cost_pct(None) > estimate_cost_pct(1e9)

    def test_breakeven_at_zero_cost_is_the_rr_identity(self):
        # p = 1/(1+rr): an edgeless system at RR 1.5 lands on exactly 40%.
        assert breakeven_win_rate(1.5, 0.0) == pytest.approx(0.4)
        assert breakeven_win_rate(1.0, 0.0) == pytest.approx(0.5)

    def test_cost_raises_the_bar(self):
        assert breakeven_win_rate(1.5, 0.10) > breakeven_win_rate(1.5, 0.0)


# ── score_ticker gating ────────────────────────────────────────────────────

def _trending_long(**overrides) -> dict:
    """Indicator set that produces a LONG dominant read in a TREND regime."""
    indicators = {
        "close": 100.0, "rsi14": 55.0, "ema9": 100.5, "ema20": 100.0, "ema50": 99.0,
        "macd": 0.4, "macd_histogram": 0.2, "atr14": 1.0, "adx14": 30.0,
        "bb_upper": 103.0, "bb_middle": 100.0, "bb_lower": 97.0, "bb_width_pct": 6.0,
        "day_high": 99.5, "day_low": 96.0, "or_high": 99.0, "or_low": 97.0,
    }
    indicators.update(overrides)
    return indicators


def _score(**kwargs) -> dict:
    params = dict(
        ticker="TEST", last=100.0, avg_dollar_volume=500_000_000.0,
        indicators=_trending_long(), phase="REGULAR",
        minutes_since_open=120.0, minutes_to_close=150.0,
        hourly_direction="LONG", daily_direction="LONG",
    )
    params.update(kwargs)
    return score_ticker(**params)


class TestScoreTickerGating:
    def test_liquid_trending_long_is_actionable(self):
        out = _score()
        assert out["dominant"] == "LONG"
        assert out["trade_signal"] in ("STRONG_BUY", "BUY_CANDIDATE")
        assert out["cost_ratio"] <= _MAX_COST_RATIO

    def test_illiquid_name_is_cost_vetoed(self):
        """A thin name's estimated spread eats too much of the stop to be tradeable,
        regardless of how good the setup looks."""
        out = _score(avg_dollar_volume=400_000.0, min_avg_dollar_volume=0.0)
        assert out["cost_ratio"] > _MAX_COST_RATIO
        assert out["trade_signal"] == "AVOID"
        assert "cost" in out["signal_reason"].lower()

    def test_model_veto_downgrades_to_watch_only(self):
        confident = _score(model_prob=0.95)
        vetoed = _score(model_prob=0.05)
        assert confident["trade_signal"] in ("STRONG_BUY", "BUY_CANDIDATE")
        assert vetoed["trade_signal"] == "WATCH_ONLY"
        assert "P(win)" in vetoed["signal_reason"]

    def test_required_prob_sits_above_cost_adjusted_breakeven(self):
        out = _score()
        expected = breakeven_win_rate(_RR, out["cost_ratio"]) + _PROB_MARGIN
        assert out["required_prob"] == pytest.approx(expected, abs=1e-4)

    def test_model_never_promotes_a_setup_the_rules_rejected(self):
        """The model is a veto, not a promoter — a 99% probability on a no-setup
        read must not manufacture a signal."""
        flat = {
            "close": 100.0, "rsi14": 50.0, "ema9": 100.0, "ema20": 100.0,
            "macd": 0.0, "macd_histogram": 0.0, "atr14": 1.0, "adx14": 20.0,
            "bb_upper": 103.0, "bb_middle": 100.0, "bb_lower": 97.0,
            "day_high": 101.0, "day_low": 99.0,
        }
        out = _score(indicators=flat, model_prob=0.99)
        assert out["trade_signal"] in ("AVOID", "WATCH_ONLY")

    def test_strength_bonus_is_inside_total_score(self):
        """RS has to be folded into the score that drives the decision, not added
        afterwards — otherwise the displayed score and the trained score diverge."""
        plain = _score()
        boosted = _score(strength_bonus=10.0)
        assert boosted["total_score"] == pytest.approx(plain["total_score"] + 10.0)

    def test_provisional_levels_exist_even_when_not_actionable(self):
        """Feature extraction needs a real stop size even for a vetoed setup;
        otherwise stop_pct trains as a zero."""
        out = _score(model_prob=0.01)
        assert out["trade_signal"] == "WATCH_ONLY"
        assert out["suggested_stop"] is None
        assert out["prov_stop_pct"] > 0


# ── Features ───────────────────────────────────────────────────────────────

class TestFeatures:
    BASE = {
        "close": 100.0, "atr14": 1.0, "rsi14": 70.0, "ema9": 100.5, "ema20": 100.0,
        "ema50": 99.0, "macd_histogram": 0.2, "adx14": 30.0, "bb_width_pct": 6.0,
        "stop_pct": 2.5, "cost_pct": 0.03, "total_score": 70.0,
        "momentum_score": 40.0, "reversion_score": 0.0, "breakout_score": 10.0,
        "mtf_score": 30.0, "sr_score": 15.0,
        "nearest_support": 98.0, "nearest_resistance": 104.0,
        "hourly_direction": "LONG", "daily_direction": "LONG",
        "rs_vs_spy": 1.5, "day_change_pct": 2.0, "day_high": 101.0, "day_low": 97.0,
        "rel_volume": 2.0, "avg_dollar_volume": 100_000_000.0,
        "market_phase": "REGULAR", "as_of": "2026-07-28T14:30:00+00:00",
    }

    def test_contract_is_complete_and_ordered(self):
        feats = build_features(self.BASE, 1)
        assert set(feats) == set(FEATURE_NAMES)
        assert to_vector(feats) == [feats[n] for n in FEATURE_NAMES]

    def test_every_value_is_finite_even_from_an_empty_snapshot(self):
        for direction in (1, -1):
            for value in build_features({}, direction).values():
                assert isinstance(value, float)
                assert math.isfinite(value)

    def test_direction_relative_features_flip_with_direction(self):
        long_f = build_features(self.BASE, 1)
        short_f = build_features(self.BASE, -1)
        # RSI 70 supports a long, opposes a short
        assert long_f["rsi_dir"] == pytest.approx(70.0)
        assert short_f["rsi_dir"] == pytest.approx(30.0)
        for name in ("ema_gap_atr", "macd_hist_atr", "rs_dir", "day_change_dir"):
            assert long_f[name] == pytest.approx(-short_f[name])
        assert long_f["hourly_agrees"] == 1.0
        assert short_f["hourly_agrees"] == -1.0

    def test_structure_is_read_from_the_trade_s_point_of_view(self):
        long_f = build_features(self.BASE, 1)
        short_f = build_features(self.BASE, -1)
        # Long: resistance (104) is the target-side level, support (98) protective.
        assert long_f["dist_to_target_level_atr"] == pytest.approx(4.0)
        assert long_f["dist_to_protective_level_atr"] == pytest.approx(2.0)
        # Short: the roles swap.
        assert short_f["dist_to_target_level_atr"] == pytest.approx(2.0)
        assert short_f["dist_to_protective_level_atr"] == pytest.approx(4.0)

    def test_atr_is_normalised_so_it_cannot_identify_the_ticker(self):
        cheap = build_features({**self.BASE, "close": 10.0, "atr14": 0.1}, 1)
        pricey = build_features({**self.BASE, "close": 1000.0, "atr14": 10.0}, 1)
        assert cheap["atr_pct"] == pytest.approx(pricey["atr_pct"])

    def test_ratios_are_clipped_against_a_near_zero_atr(self):
        feats = build_features({**self.BASE, "atr14": 1e-9}, 1)
        for name in ("ema_gap_atr", "macd_hist_atr", "extension_atr"):
            assert abs(feats[name]) <= 10.0

    def test_cost_ratio_is_cost_over_stop(self):
        feats = build_features({**self.BASE, "cost_pct": 0.25, "stop_pct": 1.0}, 1)
        assert feats["cost_ratio"] == pytest.approx(0.25)

    def test_session_flags(self):
        first = build_features({**self.BASE, "as_of": "2026-07-28T13:45:00+00:00"}, 1)
        mid = build_features({**self.BASE, "as_of": "2026-07-28T17:00:00+00:00"}, 1)
        last = build_features({**self.BASE, "as_of": "2026-07-28T19:45:00+00:00"}, 1)
        assert (first["first_hour"], first["last_hour"]) == (1.0, 0.0)
        assert (mid["first_hour"], mid["last_hour"]) == (0.0, 0.0)
        assert (last["first_hour"], last["last_hour"]) == (0.0, 1.0)
        assert first["session_progress"] < mid["session_progress"] < last["session_progress"]


# ── Time helpers ───────────────────────────────────────────────────────────

class TestTimeUtil:
    def test_parses_both_sources(self):
        # yfinance client ISO, and a bare SQLite CURRENT_TIMESTAMP
        assert parse_ts("2026-07-28T14:30:00+00:00") is not None
        assert parse_ts("2026-07-28 14:30:00") is not None
        assert parse_ts("2026-07-28T14:30:00.123456789Z") is not None
        assert parse_ts("") is None
        assert parse_ts("not a date") is None

    def test_naive_input_is_treated_as_utc(self):
        aware = parse_ts("2026-07-28T14:30:00+00:00")
        naive = parse_ts("2026-07-28 14:30:00")
        assert aware == naive

    def test_minutes_since_open_is_keyed_to_the_bar_s_own_date(self):
        # 13:30 UTC = 09:30 ET during EDT
        assert minutes_since_open_at("2026-07-28T13:30:00+00:00") == pytest.approx(0.0)
        assert minutes_since_open_at("2026-07-28T14:30:00+00:00") == pytest.approx(60.0)

    def test_session_progress_is_clipped_outside_regular_hours(self):
        assert session_progress("2026-07-28T09:00:00+00:00") == 0.0   # pre-market
        assert session_progress("2026-07-28T23:00:00+00:00") == 1.0   # after close


# ── Relative volume ────────────────────────────────────────────────────────

class TestRelativeVolume:
    def test_measures_last_bar_against_the_prior_average(self):
        bars = [{"volume": 100} for _ in range(20)] + [{"volume": 300}]
        assert calculate_relative_volume(bars) == pytest.approx(3.0)

    def test_handles_missing_or_zero_volume(self):
        assert calculate_relative_volume([]) is None
        assert calculate_relative_volume([{"volume": 0}, {"volume": 0}]) is None


# ── Model ──────────────────────────────────────────────────────────────────

def _separable_dataset(n: int = 400, seed: int = 7):
    """A dataset with a real, learnable signal in the first feature."""
    rng = np.random.default_rng(seed)
    k = len(FEATURE_NAMES)
    X = rng.normal(size=(n, k))
    logits = 1.4 * X[:, 1] - 0.8 * X[:, 2]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    # Winners pay 1.5R, losers lose 1R — the geometry the scorer actually uses.
    r = np.where(y == 1, 1.5, -1.0)
    return X, y, r


class TestModel:
    def test_fit_then_predict_returns_probabilities(self):
        X, y, _ = _separable_dataset()
        model = fit(X, y)
        for row in X[:20]:
            p = model.predict_proba_vector(row)
            assert 0.0 <= p <= 1.0

    def test_learns_a_real_signal(self):
        X, y, _ = _separable_dataset()
        model = fit(X, y)
        preds = [model.predict_proba_vector(row) for row in X]
        assert roc_auc(y, preds) > 0.7

    def test_json_round_trip_is_exact(self):
        X, y, _ = _separable_dataset(n=200)
        model = fit(X, y)
        restored = StockModel.from_json(model.to_json())
        assert restored.feature_version == FEATURE_VERSION
        assert restored.feature_names == list(FEATURE_NAMES)
        for row in X[:20]:
            assert restored.predict_proba_vector(row) == pytest.approx(
                model.predict_proba_vector(row)
            )

    def test_predict_proba_tolerates_a_partial_feature_dict(self):
        X, y, _ = _separable_dataset(n=200)
        model = fit(X, y)
        assert 0.0 <= model.predict_proba({"direction": 1.0}) <= 1.0

    def test_auc_handles_ties_and_single_class(self):
        assert roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
        assert roc_auc([1, 1, 1], [0.2, 0.6, 0.9]) is None
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_brier_and_top_decile(self):
        assert brier([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
        y = [0] * 10 + [1] * 10
        p = [0.1] * 10 + [0.9] * 10
        assert top_decile_precision(y, p) == pytest.approx(1.0)
        assert top_decile_precision([1, 0], [0.9, 0.1]) is None  # too few rows

    def test_calibration_bins_cover_every_row(self):
        X, y, _ = _separable_dataset(n=300)
        model = fit(X, y)
        p = [model.predict_proba_vector(row) for row in X]
        bins = calibration_bins(y, p, bins=5)
        assert sum(b["n"] for b in bins) == len(y)

    def test_walk_forward_is_out_of_sample(self):
        X, y, r = _separable_dataset(n=400)
        report = walk_forward(X, y, r, folds=4, min_train=150)
        assert "error" not in report
        # Every scored row is a prediction from a model that never saw it.
        assert len(report["oos_p"]) == 400 - 150
        assert report["auc"] > 0.6
        assert len(report["folds"]) == 4

    def test_walk_forward_refuses_when_there_is_nothing_to_test_on(self):
        X, y, r = _separable_dataset(n=120)
        report = walk_forward(X, y, r, folds=5, min_train=MIN_TRAIN_SAMPLES)
        assert "error" in report

    def test_gated_performance_selects_and_scores(self):
        p = [0.2, 0.4, 0.6, 0.8]
        r = [-1.0, -1.0, 1.5, 1.5]
        gated = gated_performance(p, r, 0.5)
        assert gated["trades_taken"] == 2
        assert gated["expectancy_r"] == pytest.approx(1.5)
        assert gated["baseline_expectancy_r"] == pytest.approx(0.25)

    def test_gated_performance_with_no_qualifying_trades(self):
        gated = gated_performance([0.1, 0.2], [1.0, 1.0], 0.9)
        assert gated["trades_taken"] == 0
        assert gated["expectancy_r"] is None

    def test_build_matrix_maps_outcomes_to_labels(self):
        rows = [
            {"features": build_features(TestFeatures.BASE, 1), "outcome": "WIN", "r_multiple": 1.5},
            {"features": build_features(TestFeatures.BASE, -1), "outcome": "LOSS", "r_multiple": -1.0},
            {"features": build_features(TestFeatures.BASE, 1), "outcome": "BREAKEVEN", "r_multiple": None},
        ]
        X, y, r = build_matrix(rows)
        assert X.shape == (3, len(FEATURE_NAMES))
        assert list(y) == [1.0, 0.0, 0.0]
        assert list(r) == [1.5, -1.0, 0.0]


# ── Storage: the loop that produces training data ──────────────────────────

@pytest.fixture()
def storage(tmp_path) -> Storage:
    return Storage(tmp_path / "test.sqlite3")


def _bars(prices, start_hour: int = 14):
    """5m bars whose high/low bracket each price, on a single session."""
    out = []
    for i, px in enumerate(prices):
        out.append({
            "timestamp": f"2026-07-28T{start_hour:02d}:{i * 5:02d}:00+00:00",
            "open": px, "high": px + 0.05, "low": px - 0.05,
            "close": px, "volume": 1000,
        })
    return out


class TestTrackingAndEvaluation:
    def test_armed_signal_stores_its_feature_vector(self, storage):
        feats = build_features(TestFeatures.BASE, 1)
        tid = storage.record_tracked_signal(
            ticker="AAPL", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
            features=feats, feature_version=FEATURE_VERSION,
            model_prob=0.55, required_prob=0.45, cost_pct=0.03, cost_ratio=0.012,
            total_score=72.0, adx14=30.0, regime="TREND",
            market_phase="REGULAR", rs_vs_spy=1.5,
        )
        assert tid is not None
        row = storage.load_tracked_signals("open")[0]
        assert json.loads(row["features_json"]) == feats
        assert row["feature_version"] == FEATURE_VERSION

    def test_model_mode_is_persisted_at_arm_time(self, storage):
        storage.record_tracked_signal(
            ticker="MSFT", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
            model_prob=0.55, required_prob=0.44, model_mode="shadow",
        )
        row = storage.load_tracked_signals("open")[0]
        # Without this, shadow rows and gated rows pool together in the report and
        # the censored sample silently flatters the model.
        assert row["model_mode"] == "shadow"

    def test_dedupe_suppresses_a_rearm(self, storage):
        kwargs = dict(
            ticker="AAPL", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
        )
        assert storage.record_tracked_signal(**kwargs) is not None
        assert storage.record_tracked_signal(**kwargs) is None

    def test_target_touch_resolves_as_a_win_net_of_cost(self, storage):
        storage.record_tracked_signal(
            ticker="AAPL", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
            features=build_features(TestFeatures.BASE, 1),
            feature_version=FEATURE_VERSION, cost_pct=0.10,
        )
        resolved = storage.evaluate_tracked_signals("AAPL", _bars([100.0, 101.0, 104.0]))
        assert resolved == 1

        outcome = storage.load_trade_outcomes()[0]
        assert outcome["outcome"] == "WIN"
        assert outcome["exit_reason"] == "TARGET"
        assert outcome["gross_dollars"] == pytest.approx(3.75)
        # 0.10% of a $100 entry = $0.10 round trip
        assert outcome["cost_dollars"] == pytest.approx(0.10)
        assert outcome["net_dollars"] == pytest.approx(3.65)
        # R is computed from net, not gross — otherwise every result is overstated.
        assert outcome["r_multiple"] == pytest.approx(3.65 / 2.5, abs=0.01)

    def test_stop_is_checked_before_target_within_a_bar(self, storage):
        """A bar that straddles both levels is conservatively a loss — the
        alternative silently inflates the win rate."""
        storage.record_tracked_signal(
            ticker="AAPL", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=99.0, target=101.0,
            stop_dollars=1.0, target_dollars=1.0, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
        )
        straddle = [{
            "timestamp": "2026-07-28T14:05:00+00:00",
            "open": 100.0, "high": 101.5, "low": 98.5, "close": 100.0, "volume": 1000,
        }]
        storage.evaluate_tracked_signals("AAPL", straddle)
        assert storage.load_trade_outcomes()[0]["outcome"] == "LOSS"

    def test_short_resolves_on_the_mirrored_levels(self, storage):
        storage.record_tracked_signal(
            ticker="AAPL", signal="SHORT_CANDIDATE", direction=-1,
            entry=100.0, stop=102.5, target=96.25,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
        )
        storage.evaluate_tracked_signals("AAPL", _bars([100.0, 98.0, 96.0]))
        outcome = storage.load_trade_outcomes()[0]
        assert outcome["outcome"] == "WIN"
        assert outcome["exit_reason"] == "TARGET"

    def test_unresolved_signal_stays_open(self, storage):
        storage.record_tracked_signal(
            ticker="AAPL", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
        )
        assert storage.evaluate_tracked_signals("AAPL", _bars([100.0, 100.5, 101.0])) == 0
        assert len(storage.load_tracked_signals("open")) == 1

    def test_hold_minutes_measures_the_trade_not_the_scan_gap(self, storage):
        storage.record_tracked_signal(
            ticker="AAPL", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
        )
        # Bars at 14:00, 14:05, 14:10 — the target is hit on the third, 10 min in.
        storage.evaluate_tracked_signals("AAPL", _bars([100.0, 101.0, 104.0]))
        assert storage.load_trade_outcomes()[0]["hold_minutes"] == 10

    def test_only_forward_bars_can_resolve_a_signal(self, storage):
        """Bars at or before the entry timestamp must not resolve the trade, or the
        loop would grade itself on data it already had."""
        storage.record_tracked_signal(
            ticker="AAPL", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:10:00+00:00",
        )
        # These bars all precede the entry, including one that would have hit target.
        assert storage.evaluate_tracked_signals("AAPL", _bars([100.0, 104.0, 105.0])) == 0


class TestTrainingRows:
    def _resolve_one(self, storage, ticker: str, win: bool, direction: int = 1):
        storage.record_tracked_signal(
            ticker=ticker, signal="BUY_CANDIDATE" if direction == 1 else "SHORT_CANDIDATE",
            direction=direction, entry=100.0,
            stop=97.5 if direction == 1 else 102.5,
            target=103.75 if direction == 1 else 96.25,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
            features=build_features(TestFeatures.BASE, direction),
            feature_version=FEATURE_VERSION, cost_pct=0.03,
        )
        hit = 104.0 if (win == (direction == 1)) else 96.0
        storage.evaluate_tracked_signals(ticker, _bars([100.0, hit]))

    def test_resolved_trades_become_training_rows(self, storage):
        self._resolve_one(storage, "AAPL", win=True)
        self._resolve_one(storage, "MSFT", win=False)
        rows = storage.load_training_rows(feature_version=FEATURE_VERSION)
        assert len(rows) == 2
        assert {r["outcome"] for r in rows} == {"WIN", "LOSS"}
        assert all(set(r["features"]) == set(FEATURE_NAMES) for r in rows)

    def test_rows_without_stored_features_are_excluded_not_imputed(self, storage):
        """Legacy trades predate feature logging; their inputs are gone. Training on
        imputed values would produce an AUC that means nothing."""
        storage.record_tracked_signal(
            ticker="LEGACY", signal="BUY_CANDIDATE", direction=1,
            entry=100.0, stop=97.5, target=103.75,
            stop_dollars=2.5, target_dollars=3.75, atr14=1.0,
            entry_ts="2026-07-28T14:00:00+00:00",
        )
        storage.evaluate_tracked_signals("LEGACY", _bars([100.0, 104.0]))
        assert storage.load_training_rows(feature_version=FEATURE_VERSION) == []

    def test_a_different_feature_version_is_filtered_out(self, storage):
        self._resolve_one(storage, "AAPL", win=True)
        assert storage.load_training_rows(feature_version=FEATURE_VERSION + 99) == []

    def test_rows_come_back_oldest_first(self, storage):
        for t in ("A", "B", "C"):
            self._resolve_one(storage, t, win=True)
        rows = storage.load_training_rows(feature_version=FEATURE_VERSION)
        ids = [r["tracking_id"] for r in rows]
        assert ids == sorted(ids)


class TestModelStore:
    METRICS = {
        "algo": "logistic_l2", "feature_version": FEATURE_VERSION,
        "n_train": 200, "n_test": 80, "auc": 0.61, "brier": 0.23,
        "top_decile_prec": 0.55, "base_rate": 0.41,
    }

    def test_saved_candidate_is_inactive_until_promoted(self, storage):
        model_id = storage.save_model('{"algo":"x"}', self.METRICS, activate=False)
        assert storage.load_active_model_json() is None
        assert storage.activate_model(model_id) is True
        assert storage.load_active_model_json() == '{"algo":"x"}'

    def test_promotion_is_exclusive_and_reversible(self, storage):
        first = storage.save_model('{"n":1}', self.METRICS, activate=True)
        second = storage.save_model('{"n":2}', self.METRICS, activate=True)
        assert storage.load_active_model_json() == '{"n":2}'

        # Rollback to the earlier model.
        assert storage.activate_model(first) is True
        assert storage.load_active_model_json() == '{"n":1}'
        assert sum(1 for m in storage.load_models() if m["is_active"]) == 1
        assert second != first

    def test_deactivate_all_reverts_to_rules_only(self, storage):
        storage.save_model('{"n":1}', self.METRICS, activate=True)
        storage.deactivate_all_models()
        assert storage.load_active_model_json() is None
        assert len(storage.load_models()) == 1  # kept, not deleted

    def test_activating_an_unknown_id_fails_cleanly(self, storage):
        assert storage.activate_model(999) is False

    def test_round_trip_through_the_store_preserves_predictions(self, storage):
        X, y, _ = _separable_dataset(n=200)
        model = fit(X, y)
        storage.save_model(model.to_json(), self.METRICS, activate=True)
        served = StockModel.from_json(storage.load_active_model_json())
        for row in X[:20]:
            assert served.predict_proba_vector(row) == pytest.approx(
                model.predict_proba_vector(row)
            )
