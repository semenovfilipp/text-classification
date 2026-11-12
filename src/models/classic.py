from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_model(spec, seed=42):
    kind = spec["type"]
    params = dict(spec.get("params", {}))
    if kind == "logreg":
        model = LogisticRegression(max_iter=params.pop("max_iter", 2000), random_state=seed, **params)
        return make_pipeline(StandardScaler(), model) if spec.get("scale") else model
    if kind == "linear_svc":
        return LinearSVC(random_state=seed, **params)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False, **params)
    raise ValueError(kind)


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
