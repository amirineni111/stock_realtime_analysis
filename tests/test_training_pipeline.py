"""
End-to-end coverage of the retrain pipeline: stored rows → walk-forward evaluation
→ quality gate → candidate → promotion → the scanner serving that model.

The gate is the part worth testing hardest. Its whole job is to refuse a model with
no demonstrated edge, and a gate that passes everything is worse than no gate — it
launders noise into a veto that suppresses real setups.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from stocks.features import FEATURE_NAMES, FEATURE_VERSION
from stocks.model import MIN_TRAIN_SAMPLES, StockModel
from stocks.storage import Storage
from stocks.training import evaluate_and_fit, gate_summary, save_candidate


@pytest.fixture()
def storage(tmp_path) -> Storage:
    return Storage(tmp_path / "train.sqlite3")


def _seed(storage: Storage, n: int, signal_strength: float, seed: int = 11) -> None:
    """
    Insert ``n`` resolved trades whose outcome depends on the features with the given
    ``signal_strength``. At 0.0 the labels are pure coin flips, which is what a model
    with no edge must be shown to fail on.

    Rows are written directly rather than through ``record_tracked_signal`` so the
    dedupe/cooldown rule doesn't suppress a synthetic burst of trades on one ticker.
    """
    rng = np.random.default_rng(seed)
    k = len(FEATURE_NAMES)
    with storage._connect() as conn:
        for i in range(n):
            values = rng.normal(size=k)
            feats = {name: float(v) for name, v in zip(FEATURE_NAMES, values)}
            logit = signal_strength * (values[1] - values[2])
            win = rng.uniform() < 1.0 / (1.0 + np.exp(-logit))
            # Winners pay 1.5R, losers lose 1R — the bracket the scorer actually uses.
            r_multiple = 1.5 if win else -1.0
            created = f"2026-07-{1 + i // 40:02d}T{10 + i % 6:02d}:{i % 60:02d}:00+00:00"

            cur = conn.execute(
                "INSERT INTO stock_signal_tracking "
                "(ticker,signal,direction,entry_price,stop_price,target_price,"
                "stop_dollars,target_dollars,atr14,entry_ts,status,created_at,"
                "features_json,feature_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,'closed',?,?,?)",
                (f"T{i % 9}", "BUY_CANDIDATE", 1, 100.0, 97.5, 103.75,
                 2.5, 3.75, 1.0, created, created,
                 json.dumps(feats), FEATURE_VERSION),
            )
            conn.execute(
                "INSERT INTO stock_trade_outcomes "
                "(watchlist_id,tracking_id,ticker,signal,entry_price,exit_price,"
                "exit_dollars,r_multiple,outcome,net_dollars,exit_reason) "
                "VALUES (0,?,?,?,?,?,?,?,?,?,?)",
                (cur.lastrowid, f"T{i % 9}", "BUY_CANDIDATE", 100.0,
                 103.75 if win else 97.5, r_multiple * 2.5, r_multiple,
                 "WIN" if win else "LOSS", r_multiple * 2.5,
                 "TARGET" if win else "STOP"),
            )


class TestEvaluateAndFit:
    def test_refuses_below_the_minimum_sample(self, storage):
        _seed(storage, 30, signal_strength=2.0)
        report = evaluate_and_fit(storage)
        assert report["error"] is not None
        assert report["model"] is None
        assert str(MIN_TRAIN_SAMPLES) in report["error"]
        assert "Not enough data" in gate_summary(report)

    def test_refuses_on_an_empty_database(self, storage):
        report = evaluate_and_fit(storage)
        assert report["error"] is not None
        assert report["n_rows"] == 0

    def test_learns_and_passes_on_a_real_signal(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        report = evaluate_and_fit(storage, folds=4)
        assert report["error"] is None
        assert report["metrics"]["auc"] > 0.6
        assert report["passes"] is True
        assert gate_summary(report).startswith("PASS")

    def test_gate_rejects_a_model_with_no_edge(self, storage):
        """Coin-flip labels: AUC lands near 0.50 and no threshold turns a profit."""
        _seed(storage, 400, signal_strength=0.0)
        report = evaluate_and_fit(storage, folds=4)
        assert report["error"] is None
        assert report["passes"] is False
        assert gate_summary(report).startswith("FAIL")

    def test_gate_respects_a_raised_auc_bar(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        assert evaluate_and_fit(storage, folds=4, min_auc=0.99)["passes"] is False

    def test_metrics_are_out_of_sample_and_complete(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        report = evaluate_and_fit(storage, folds=4)
        m = report["metrics"]
        for key in ("algo", "feature_version", "n_train", "n_test", "auc", "brier",
                    "top_decile_prec", "base_rate", "best_threshold"):
            assert key in m
        assert m["feature_version"] == FEATURE_VERSION
        assert m["n_train"] == 400
        # Tested on strictly fewer rows than trained on — the walk forward holds
        # back the first window.
        assert 0 < m["n_test"] < m["n_train"]
        assert len(report["folds"]) == 4
        assert report["fold_auc_std"] is not None

    def test_reports_expectancy_at_several_thresholds(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        report = evaluate_and_fit(storage, folds=4)
        assert len(report["gated"]) >= 3
        for g in report["gated"]:
            assert g["trades_taken"] <= g["trades_available"]
        # A real edge means gating beats taking everything.
        assert report["best"]["expectancy_r"] > report["baseline_expectancy_r"]

    def test_calibration_and_coefficients_are_reported(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        report = evaluate_and_fit(storage, folds=4)
        assert sum(b["n"] for b in report["calibration"]) == report["metrics"]["n_test"]
        names = [name for name, _ in report["coefficients"]]
        assert set(names).issubset(set(FEATURE_NAMES))

    def test_nothing_is_written_to_the_database(self, storage):
        """evaluate_and_fit evaluates; persistence is the caller's decision."""
        _seed(storage, 400, signal_strength=2.0)
        evaluate_and_fit(storage, folds=4)
        assert storage.load_models() == []
        assert storage.load_active_model_json() is None


class TestCandidateAndPromotion:
    def test_candidate_is_saved_inactive(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        report = evaluate_and_fit(storage, folds=4)
        model_id = save_candidate(storage, report, notes="unit test")

        assert model_id is not None
        assert storage.load_active_model_json() is None, "saving must not promote"
        stored = storage.load_models()[0]
        assert stored["is_active"] == 0
        assert stored["notes"] == "unit test"
        assert stored["auc"] == pytest.approx(report["metrics"]["auc"])

    def test_failed_report_saves_nothing(self, storage):
        _seed(storage, 30, signal_strength=2.0)
        report = evaluate_and_fit(storage)
        assert save_candidate(storage, report) is None
        assert storage.load_models() == []

    def test_shadow_model_is_served_but_not_active(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        mid = save_candidate(storage, evaluate_and_fit(storage, folds=4))
        assert storage.shadow_model(mid) is True
        # Shadow is the offline lane: the scanner can score with it, but nothing
        # gates on it, so load_active_model_json must stay empty.
        assert storage.load_active_model_json() is None
        assert storage.load_shadow_model_json() is not None

    def test_a_model_cannot_gate_and_shadow_at_once(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        mid = save_candidate(storage, evaluate_and_fit(storage, folds=4))
        storage.shadow_model(mid)
        storage.activate_model(mid)
        row = [m for m in storage.load_models() if m["id"] == mid][0]
        assert (row["is_active"], row["is_shadow"]) == (1, 0)
        # And back the other way, so promotion is reversible into shadow.
        storage.shadow_model(mid)
        row = [m for m in storage.load_models() if m["id"] == mid][0]
        assert (row["is_active"], row["is_shadow"]) == (0, 1)

    def test_only_one_model_shadows_at_a_time(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        first = save_candidate(storage, evaluate_and_fit(storage, folds=4))
        second = save_candidate(storage, evaluate_and_fit(storage, folds=4, l2=5.0))
        storage.shadow_model(first)
        storage.shadow_model(second)
        assert [m["id"] for m in storage.load_models() if m["is_shadow"]] == [second]

    def test_shadowing_a_missing_id_is_rejected(self, storage):
        assert storage.shadow_model(99999) is False

    def test_clear_shadow_leaves_the_model_stored(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        mid = save_candidate(storage, evaluate_and_fit(storage, folds=4))
        storage.shadow_model(mid)
        storage.clear_shadow_model()
        assert storage.load_shadow_model_json() is None
        assert any(m["id"] == mid for m in storage.load_models())

    def test_promoted_model_serves_the_same_predictions(self, storage):
        _seed(storage, 400, signal_strength=2.0)
        report = evaluate_and_fit(storage, folds=4)
        model_id = save_candidate(storage, report)
        storage.activate_model(model_id)

        served = StockModel.from_json(storage.load_active_model_json())
        trained = report["model"]
        probe = {name: 0.5 for name in FEATURE_NAMES}
        assert served.predict_proba(probe) == pytest.approx(trained.predict_proba(probe))
        assert served.feature_version == FEATURE_VERSION


class TestScannerServing:
    """The scanner must fail closed rather than serve a mismatched model."""

    def test_stale_feature_version_is_refused(self, storage):
        from stocks.scanner import _load_model

        model = StockModel(
            coefficients=[0.0] * len(FEATURE_NAMES), intercept=0.0,
            mean=[0.0] * len(FEATURE_NAMES), std=[1.0] * len(FEATURE_NAMES),
            feature_version=FEATURE_VERSION + 1,
        )
        storage.save_model(model.to_json(), {"feature_version": FEATURE_VERSION + 1},
                           activate=True)
        assert _load_model(storage) is None

    def test_current_feature_version_is_served(self, storage):
        from stocks.scanner import _load_model

        model = StockModel(
            coefficients=[0.0] * len(FEATURE_NAMES), intercept=0.0,
            mean=[0.0] * len(FEATURE_NAMES), std=[1.0] * len(FEATURE_NAMES),
        )
        storage.save_model(model.to_json(), {"feature_version": FEATURE_VERSION},
                           activate=True)
        loaded = _load_model(storage)
        assert loaded is not None
        assert loaded.predict_proba({}) == pytest.approx(0.5)

    def test_no_model_means_rules_only(self, storage):
        from stocks.scanner import _load_model
        assert _load_model(storage) is None

    def test_a_shadow_model_is_only_served_to_the_shadow_lane(self, storage):
        from stocks.scanner import _load_model

        model = StockModel(
            coefficients=[0.0] * len(FEATURE_NAMES), intercept=0.0,
            mean=[0.0] * len(FEATURE_NAMES), std=[1.0] * len(FEATURE_NAMES),
        )
        mid = storage.save_model(model.to_json(), {"feature_version": FEATURE_VERSION},
                                 activate=False)
        storage.shadow_model(mid)
        # If this leaked into the gating lane the scanner would veto on a model the
        # user explicitly chose not to promote.
        assert _load_model(storage) is None
        assert _load_model(storage, shadow=True) is not None

    def test_corrupt_model_json_does_not_crash_the_scan(self, storage):
        from stocks.scanner import _load_model
        storage.save_model("{not json", {"feature_version": FEATURE_VERSION}, activate=True)
        assert _load_model(storage) is None
