"""Entraînement, évaluation et interprétation du projet ChurnScope."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
THRESHOLD = 0.5
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
YES_NO = {"Yes", "No"}
INTERNET_OPTIONS = {"Yes", "No", "No internet service"}
INTERNET_FEATURES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]
ALLOWED_CATEGORIES = {
    "gender": {"Female", "Male"}, "SeniorCitizen": {0, 1},
    "Partner": YES_NO, "Dependents": YES_NO, "PhoneService": YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{column: INTERNET_OPTIONS for column in INTERNET_FEATURES},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"},
    "Churn": YES_NO,
}


def prepare_directories() -> None:
    for folder in (PROCESSED, REPORTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str, **kwargs) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180, **kwargs)
    plt.close()


def audit_raw_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Vérifie le schéma et les règles simples avant toute modélisation."""
    if set(raw.columns) != set(EXPECTED_COLUMNS):
        missing = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
        extra = sorted(set(raw.columns) - set(EXPECTED_COLUMNS))
        raise ValueError(f"Colonnes absentes : {missing} ; colonnes inattendues : {extra}.")
    if raw.empty:
        raise ValueError("Le fichier de données est vide.")
    if raw.drop(columns=["TotalCharges"]).isna().any().any():
        raise ValueError("Une valeur manque hors de TotalCharges.")
    for column in raw.select_dtypes(include=["object", "string"]):
        if column != "TotalCharges" and raw[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"La colonne {column} contient une chaîne vide.")
    if raw["customerID"].duplicated().any():
        raise ValueError("Des identifiants clients sont en double.")
    for column, allowed in ALLOWED_CATEGORIES.items():
        unexpected = set(raw[column].unique()) - allowed
        if unexpected:
            raise ValueError(f"Catégories inattendues dans {column} : {sorted(map(str, unexpected))}.")
    if set(raw["Churn"].unique()) != YES_NO:
        raise ValueError("Churn doit contenir les deux classes Yes et No.")
    if not pd.api.types.is_numeric_dtype(raw["tenure"]) or not pd.api.types.is_numeric_dtype(raw["MonthlyCharges"]):
        raise ValueError("tenure et MonthlyCharges doivent être numériques.")
    if raw["tenure"].lt(0).any() or raw["tenure"].mod(1).ne(0).any():
        raise ValueError("tenure doit être un nombre entier positif ou nul.")
    if raw["MonthlyCharges"].lt(0).any():
        raise ValueError("MonthlyCharges ne peut pas être négatif.")
    if raw["PhoneService"].eq("No").ne(raw["MultipleLines"].eq("No phone service")).any():
        raise ValueError("PhoneService et MultipleLines sont incohérents.")
    for column in INTERNET_FEATURES:
        if raw["InternetService"].eq("No").ne(raw[column].eq("No internet service")).any():
            raise ValueError(f"InternetService et {column} sont incohérents.")

    charge_text = raw["TotalCharges"].astype("string")
    blank_charges = charge_text.str.strip().eq("").fillna(True)
    parsed_charges = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    if (parsed_charges.isna() & ~blank_charges).any():
        raise ValueError("TotalCharges contient du texte non numérique non vide.")
    if (parsed_charges.isna() & raw["tenure"].ne(0)).any():
        raise ValueError("TotalCharges manque pour un client déjà abonné : à examiner.")
    if parsed_charges.lt(0).any():
        raise ValueError("TotalCharges ne peut pas être négatif.")

    columns = []
    for column in EXPECTED_COLUMNS:
        values = raw[column]
        item = {
            "name": column, "dtype_raw": str(values.dtype),
            "distinct_values": int(values.nunique(dropna=True)),
            "null_values": int(values.isna().sum()),
            "blank_strings": int(values.astype("string").str.strip().eq("").fillna(False).sum()),
        }
        if column in ALLOWED_CATEGORIES:
            item["categories"] = {str(k): int(v) for k, v in values.value_counts().items()}
        columns.append(item)
    audit = {
        "rows": len(raw), "column_count": len(raw.columns),
        "duplicate_customer_ids": int(raw["customerID"].duplicated().sum()),
        "identical_profiles_without_id": int(raw.drop(columns=["customerID"]).duplicated().sum()),
        "empty_total_charges_filled_with_zero": int(parsed_charges.isna().sum()),
        "inconsistent_phone_or_internet_services": 0,
        "negative_numeric_values": 0,
        "columns": columns,
    }
    cleaned = raw.copy()
    cleaned["TotalCharges"] = parsed_charges.fillna(0)
    return cleaned, audit


def write_data_audit(audit: dict) -> None:
    (REPORTS / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Contrôle des données — ChurnScope", "",
        f"- {format(audit['rows'], ',').replace(',', ' ')} clients et {audit['column_count']} colonnes (identifiant et cible inclus).",
        f"- {audit['duplicate_customer_ids']} identifiant client en double.",
        f"- {audit['identical_profiles_without_id']} profils identiques hors identifiant : conservés, car des clients distincts peuvent avoir les mêmes caractéristiques.",
        f"- {audit['empty_total_charges_filled_with_zero']} valeurs vides de `TotalCharges` pour des clients avec `tenure = 0` : remplacées par 0.",
        "- Aucune autre valeur manquante, catégorie inattendue, valeur numérique négative ou incohérence entre services téléphoniques/Internet.",
        "", "## Vérification par colonne", "",
        "| Colonne | Type brut | Valeurs distinctes | Nulles | Chaînes vides |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in audit["columns"]:
        lines.append(f"| `{item['name']}` | {item['dtype_raw']} | {item['distinct_values']} | {item['null_values']} | {item['blank_strings']} |")
    lines += ["", "Les modalités attendues des variables catégorielles et leurs effectifs sont enregistrés dans `data_audit.json`.",
              "Ces contrôles valident ce jeu de données ; ils ne remplacent pas une revue métier sur des données nouvelles."]
    (REPORTS / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, dict]:
    """Audite, nettoie, retire l'identifiant et crée la cible 0/1."""
    raw, audit = audit_raw_data(pd.read_csv(RAW))
    customer_ids = raw["customerID"].copy()
    y = (raw["Churn"] == "Yes").astype(int)
    X = raw.drop(columns=["customerID", "Churn"])
    return X, y, customer_ids, raw, audit


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [("numeric", StandardScaler(), numerical), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        sparse_threshold=0,
    )


def metrics(y_true: pd.Series, probabilities) -> dict[str, float]:
    predictions = (probabilities >= THRESHOLD).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions)), 4),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
    }


def create_eda_figures(raw: pd.DataFrame) -> None:
    contract = pd.crosstab(raw["Contract"], raw["Churn"], normalize="index").mul(100)
    ax = contract["Yes"].sort_values(ascending=False).plot(kind="bar", color="#c75d5d")
    ax.set(title="Taux de résiliation selon le contrat", xlabel="Type de contrat", ylabel="Churn (%)", ylim=(0, 50))
    plt.xticks(rotation=0)
    save_figure("churn_by_contract.png")

    internet = pd.crosstab(raw["InternetService"], raw["Churn"], normalize="index").mul(100)
    ax = internet["Yes"].sort_values(ascending=False).plot(kind="bar", color="#4c78a8")
    ax.set(title="Taux de résiliation selon le service Internet", xlabel="Service Internet", ylabel="Churn (%)", ylim=(0, 50))
    plt.xticks(rotation=0)
    save_figure("churn_by_internet.png")

    plot_data = raw.assign(churn_label=raw["Churn"].map({"No": "Reste", "Yes": "Résilie"}))
    sns.boxplot(data=plot_data, x="churn_label", y="tenure", order=["Reste", "Résilie"], color="#8fb3c9")
    plt.title("Ancienneté selon le churn")
    plt.xlabel("")
    plt.ylabel("Ancienneté (mois)")
    save_figure("tenure_by_churn.png")

    payments = pd.crosstab(raw["PaymentMethod"], raw["Churn"], normalize="index").mul(100)
    payments["Yes"].sort_values(ascending=False).plot(kind="bar", color="#778da9", figsize=(9, 4))
    plt.title("Taux de résiliation selon le moyen de paiement")
    plt.xlabel("Moyen de paiement")
    plt.ylabel("Churn (%)")
    plt.xticks(rotation=20, ha="right")
    save_figure("churn_by_payment.png")

    sns.boxplot(data=plot_data, x="churn_label", y="MonthlyCharges", order=["Reste", "Résilie"], color="#8fb3c9")
    plt.title("Montant mensuel selon le churn")
    plt.xlabel("")
    plt.ylabel("Montant mensuel")
    save_figure("monthly_charges_by_churn.png")


def train_and_select(X_train, y_train):
    """Choisit sur validation croisée ; le jeu de test reste de côté."""
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
        scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv)
        results[name] = {"roc_auc_mean": round(float(scores.mean()), 4),
                         "roc_auc_std": round(float(scores.std()), 4)}
        print(f"{name} : ROC-AUC validation croisée = {scores.mean():.3f} ± {scores.std():.3f}", flush=True)
    best_name = max(results, key=lambda name: results[name]["roc_auc_mean"])
    best_model = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", candidates[best_name])])
    best_model.fit(X_train, y_train)
    return best_model, best_name, results


def make_error_analysis(results: pd.DataFrame) -> pd.DataFrame:
    """Décrit les erreurs par segment, sans modifier le modèle ni son seuil."""
    segments = {
        "Contrat": results["Contract"],
        "Ancienneté": pd.cut(results["tenure"], bins=[-1, 12, 24, 48, float("inf")],
                            labels=["0–12 mois", "13–24 mois", "25–48 mois", "49+ mois"]),
    }
    rows = []
    for segment_name, values in segments.items():
        for group_name, group in results.groupby(values, observed=True):
            actual = group["actual_churn"].eq(1)
            predicted = group["predicted_churn"].eq(1)
            positives = int(actual.sum())
            true_positives = int((actual & predicted).sum())
            rows.append({
                "segment": segment_name, "group": str(group_name), "clients": len(group),
                "actual_churn": positives, "detected_churn": true_positives,
                "missed_churn": int((actual & ~predicted).sum()),
                "false_alerts": int((~actual & predicted).sum()),
                "recall": round(true_positives / positives, 4) if positives else None,
            })
    return pd.DataFrame(rows)


def make_threshold_analysis(y_true: pd.Series, probabilities) -> pd.DataFrame:
    """Montre le compromis post-hoc ; aucun seuil n'est choisi sur le test."""
    rows = []
    actual = y_true.eq(1).to_numpy()
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        predicted = probabilities >= threshold
        rows.append({
            "threshold": threshold, "flagged_clients": int(predicted.sum()),
            "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(actual, predicted)), 4),
            "false_alerts": int((~actual & predicted).sum()),
            "missed_churn": int((actual & ~predicted).sum()),
        })
    return pd.DataFrame(rows)


def save_reports(model, name, cv_results, X_train, y_train, X_test, y_test, customer_ids) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
    report = {
        "best_model": name,
        "selection": {"method": "validation croisée stratifiée sur entraînement", "folds": CV_FOLDS,
                      "criterion": "roc_auc_mean", "models": cv_results},
        "test": {"selected_model": metrics(y_test, probabilities),
                 "baseline_majority_class": metrics(y_test, baseline_probabilities),
                 "confusion_matrix": dict(zip(
                     ("true_negative", "false_positive", "false_negative", "true_positive"),
                     map(int, confusion_matrix(y_test, probabilities >= THRESHOLD).ravel()),
                 ))},
        "data": {"train_rows": len(X_train), "test_rows": len(X_test),
                 "positive_rate_train": round(float(y_train.mean()), 4),
                 "positive_rate_test": round(float(y_test.mean()), 4)},
        "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "decision_threshold": THRESHOLD,
    }
    joblib.dump(model, REPORTS / "best_model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    results = X_test.copy()
    results.insert(0, "customerID", customer_ids.loc[X_test.index])
    results["actual_churn"] = y_test
    results["churn_probability"] = probabilities.round(4)
    results["predicted_churn"] = (probabilities >= THRESHOLD).astype(int)
    results.sort_values("churn_probability", ascending=False).to_csv(PROCESSED / "test_predictions.csv", index=False)
    make_error_analysis(results).to_csv(REPORTS / "error_analysis.csv", index=False)
    make_threshold_analysis(y_test, probabilities).to_csv(REPORTS / "threshold_analysis.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.plot(fpr, tpr, color="#315d85", label=f"{name} — AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="#8d98a5", label="Hasard")
    plt.title("Courbe ROC")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.legend()
    save_figure("roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.plot(recall, precision, color="#315d85", label=f"{name} — AP={average_precision_score(y_test, probabilities):.3f}")
    plt.axhline(y_test.mean(), linestyle="--", color="#8d98a5", label="Prévalence (référence)")
    plt.title("Précision selon le rappel")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.legend()
    save_figure("precision_recall_curve.png")

    ConfusionMatrixDisplay.from_predictions(y_test, (probabilities >= THRESHOLD).astype(int),
                                            display_labels=["Reste", "Résilie"], cmap="Blues")
    plt.title("Matrice de confusion sur le jeu de test")
    plt.xlabel("Prédiction")
    plt.ylabel("Observation")
    save_figure("confusion_matrix.png")

    importance = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=8, random_state=42, n_jobs=-1)
    table = pd.DataFrame({"variable": X_test.columns, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    table.to_csv(REPORTS / "feature_importance.csv", index=False)
    top = table.head(12).sort_values("importance")
    plt.barh(top["variable"], top["importance"], color="#4c78a8")
    plt.title("Variables les plus importantes")
    plt.xlabel("Baisse de l'AUC-ROC après permutation")
    save_figure("feature_importance.png")
    return report


def create_shap_report(model, X_test) -> None:
    """SHAP est optionnel : une erreur de version n'empêche pas le reste du projet."""
    try:
        import shap
        transformed = model.named_steps["preprocessor"].transform(X_test.iloc[:250])
        estimator = model.named_steps["model"]
        names = model.named_steps["preprocessor"].get_feature_names_out()
        if hasattr(estimator, "feature_importances_"):
            values = shap.TreeExplainer(estimator).shap_values(transformed)
        else:
            values = shap.LinearExplainer(estimator, transformed).shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        shap.summary_plot(values, transformed, feature_names=names, plot_type="bar", show=False, max_display=12)
        save_figure("shap_importance.png", bbox_inches="tight")
    except Exception as error:
        (REPORTS / "shap_status.txt").write_text(f"SHAP non généré : {error}", encoding="utf-8")
        print(f"Attention : SHAP non généré ({error})", flush=True)


def main() -> None:
    prepare_directories()
    X, y, customer_ids, raw, audit = load_data()
    write_data_audit(audit)
    create_eda_figures(raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    model, name, cv_results = train_and_select(X_train, y_train)
    report = save_reports(model, name, cv_results, X_train, y_train, X_test, y_test, customer_ids)
    create_shap_report(model, X_test)
    print(f"Meilleur modèle : {name}")
    print(json.dumps(report["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
