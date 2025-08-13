import os
import sys
import json
import numpy as np
import pandas as pd
import pytest


def _stock_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@pytest.fixture(scope="module", autouse=True)
def setup_env():
    prev_cwd = os.getcwd()
    stock_dir = _stock_dir()
    if stock_dir not in sys.path:
        sys.path.insert(0, stock_dir)
    os.chdir(stock_dir)
    yield
    os.chdir(prev_cwd)


def _has_xgboost():
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _load_mlpredictor():
    from psx_ai_advisor.ml_predictor import MLPredictor
    return MLPredictor


def _make_xy(n=120, p=8, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = (rng.rand(n) > 0.5).astype(int)
    return X, y


def test_xgb_param_distribution_ranges():
    MLPredictor = _load_mlpredictor()
    mlp = MLPredictor()
    if not getattr(mlp, "xgboost_available", False):
        with pytest.raises(Exception):
            mlp._get_xgb_param_distribution()
        pytest.skip("xgboost not available")
    dist = mlp._get_xgb_param_distribution()
    assert {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "gamma"}.issubset(dist.keys())
    rng = np.random.RandomState(0)
    n = dist["n_estimators"].rvs(random_state=rng)
    d = dist["max_depth"].rvs(random_state=rng)
    lr = dist["learning_rate"].rvs(random_state=rng)
    ss = dist["subsample"].rvs(random_state=rng)
    cs = dist["colsample_bytree"].rvs(random_state=rng)
    mcw = dist["min_child_weight"].rvs(random_state=rng)
    gm = dist["gamma"].rvs(random_state=rng)
    assert 50 <= int(n) <= 300
    assert 3 <= int(d) <= 10
    assert 0.01 <= float(lr) <= 0.2
    assert 0.6 <= float(ss) <= 1.0
    assert 0.6 <= float(cs) <= 1.0
    assert 1 <= int(mcw) <= 10
    assert 0.0 <= float(gm) <= 0.5


def test_optimize_xgboost_returns_estimator(monkeypatch):
    if not _has_xgboost():
        pytest.skip("xgboost not available")
    MLPredictor = _load_mlpredictor()
    from psx_ai_advisor import ml_predictor as mp

    class SmallRandomizedSearchCV:
        def __init__(self, estimator, param_distributions=None, n_iter=None, cv=None, scoring=None, random_state=None, n_jobs=None, verbose=None):
            self.estimator = estimator
            self.best_estimator_ = None
            self.best_params_ = {}
            self.best_score_ = 0.0

        def fit(self, X, y):
            self.estimator.fit(X, y)
            self.best_estimator_ = self.estimator
            self.best_params_ = {}
            self.best_score_ = 0.5
            return self

    monkeypatch.setattr(mp, "RandomizedSearchCV", SmallRandomizedSearchCV)

    mlp = MLPredictor()
    mlp.xgboost_available = True
    X, y = _make_xy(n=80, p=6)
    model = mlp._optimize_xgboost(X, y)
    import xgboost as xgb
    assert isinstance(model, xgb.XGBClassifier)


def test_calculate_ensemble_weights_normalization(monkeypatch):
    if not _has_xgboost():
        pytest.skip("xgboost not available")
    MLPredictor = _load_mlpredictor()
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    mlp = MLPredictor()
    mlp.xgboost_available = True

    def fake_cv(self, model_class, model_params, X, y, tscv, model_name="model"):
        m = {"Random Forest": 0.6, "XGBoost": 0.8}
        s = m.get(model_name, 0.5)
        return {"mean_accuracy": s, "fold_scores": [s] * self.n_splits, "cached": False}

    monkeypatch.setattr(MLPredictor, "_perform_efficient_cv_scoring", fake_cv)

    rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
    xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=0, n_estimators=10, verbosity=0)
    X, y = _make_xy()
    rf_w, xgb_w, scores = mlp._calculate_ensemble_weights(rf_model, xgb_model, X, y)
    assert abs(rf_w + xgb_w - 1.0) < 1e-8
    assert 0 < rf_w < 1 and 0 < xgb_w < 1
    assert scores["rf"] == 0.6 and scores["xgb"] == 0.8


def test_calculate_ensemble_weights_equal_when_zero(monkeypatch):
    if not _has_xgboost():
        pytest.skip("xgboost not available")
    MLPredictor = _load_mlpredictor()
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    mlp = MLPredictor()
    mlp.xgboost_available = True

    def fake_cv_zero(self, model_class, model_params, X, y, tscv, model_name="model"):
        return {"mean_accuracy": 0.0, "fold_scores": [0.0] * self.n_splits, "cached": False}

    monkeypatch.setattr(MLPredictor, "_perform_efficient_cv_scoring", fake_cv_zero)

    rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
    xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=0, n_estimators=10, verbosity=0)
    X, y = _make_xy()
    rf_w, xgb_w, _ = mlp._calculate_ensemble_weights(rf_model, xgb_model, X, y)
    assert pytest.approx(rf_w, 1e-9) == 0.5
    assert pytest.approx(xgb_w, 1e-9) == 0.5


def test_create_ensemble_model_and_metadata(monkeypatch):
    if not _has_xgboost():
        pytest.skip("xgboost not available")
    MLPredictor = _load_mlpredictor()
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    mlp = MLPredictor()
    mlp.xgboost_available = True

    def fake_weights(self, rf_model, xgb_model, X, y):
        return 0.3, 0.7, {"rf": 0.6, "xgb": 0.8, "rf_scores": [0.6], "xgb_scores": [0.8]}

    monkeypatch.setattr(MLPredictor, "_calculate_ensemble_weights", fake_weights)

    X, y = _make_xy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(Xs, y)
    xgbm = xgb.XGBClassifier(eval_metric="logloss", random_state=0, n_estimators=10, verbosity=0).fit(Xs, y)
    ensemble, meta = mlp._create_ensemble_model(rf, xgbm, X, y)
    from sklearn.ensemble import VotingClassifier
    assert isinstance(ensemble, VotingClassifier)
    assert ensemble.voting == "soft"
    assert list(ensemble.weights) == [0.3, 0.7]
    assert meta.get("model_type") == "Ensemble_RF_XGB"
    assert meta.get("ensemble_weights", {}).get("rf") == 0.3
    assert meta.get("ensemble_weights", {}).get("xgb") == 0.7


def test_training_and_prediction_flow_ensemble_or_fallback(tmp_path):
    MLPredictor = _load_mlpredictor()
    mlp = MLPredictor(model_type="ensemble")
    data_dir = "data"
    src = os.path.join(data_dir, "TEST.csv")
    dst = os.path.join(data_dir, "TEST_HISTORICAL_DATA.csv")
    assert os.path.exists(src)
    if not os.path.exists(dst):
        df = pd.read_csv(src)
        df.to_csv(dst, index=False)
    res = mlp.train_model("TEST", optimize_params=False)
    assert isinstance(res, dict)
    assert res.get("symbol", "").upper().endswith("_HISTORICAL_DATA")
    assert res.get("training_samples", 0) > 0
    pred = mlp.predict_movement("TEST")
    assert pred.get("symbol", "").upper().endswith("_HISTORICAL_DATA")
    assert pred.get("prediction") in {"UP", "DOWN"}
    assert 0.0 <= float(pred.get("confidence", 0.0)) <= 1.0
    if pred.get("model_type") == "Ensemble_RF_XGB":
        assert "individual_predictions" in pred
        assert "ensemble_weights" in pred


def test_error_handling_xgb_failure_fallback(monkeypatch):
    MLPredictor = _load_mlpredictor()
    from psx_ai_advisor.exceptions import XGBoostTrainingError
    mlp = MLPredictor(model_type="ensemble")

    def boom(*args, **kwargs):
        raise XGBoostTrainingError("forced failure")

    monkeypatch.setattr(MLPredictor, "_optimize_xgboost", lambda self, X, y: boom())
    res = mlp.train_model("TEST", optimize_params=True)
    assert res.get("model_type") in {"RandomForest", "Ensemble_RF_XGB"}
    assert res.get("decision_threshold") is not None
    if "xgboost_failure" in res:
        xf = res["xgboost_failure"]
        assert isinstance(xf, dict)


def test_error_handling_ensemble_creation_failure(monkeypatch):
    if not _has_xgboost():
        pytest.skip("xgboost not available")
    MLPredictor = _load_mlpredictor()
    mlp = MLPredictor(model_type="ensemble")

    def raise_ensemble(*args, **kwargs):
        from psx_ai_advisor.exceptions import EnsembleCreationError
        raise EnsembleCreationError("forced ensemble failure")

    monkeypatch.setattr(MLPredictor, "_create_ensemble_model", lambda self, rf, xgbm, X, y: raise_ensemble())
    res = mlp.train_model("TEST", optimize_params=False)
    assert res.get("model_type") in {"RandomForest", "Ensemble_RF_XGB"}


