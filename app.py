"""Dashboard Dash de ChurnScope : résultats du modèle de churn."""

import json
from pathlib import Path

import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent
with open(ROOT / "reports" / "metrics.json", encoding="utf-8") as file:
    report = json.load(file)

best_name = report["best_model"]
metrics = report["test"]["selected_model"]
baseline = report["test"]["baseline_majority_class"]
cv_results = report["selection"]["models"]
predictions = pd.read_csv(ROOT / "data" / "processed" / "test_predictions.csv")
predictions["Résiliation réelle"] = predictions["actual_churn"].map({0: "Non", 1: "Oui"})
errors = pd.read_csv(ROOT / "reports" / "error_analysis.csv")
contract_errors = errors.loc[errors["segment"] == "Contrat"]
thresholds = pd.read_csv(ROOT / "reports" / "threshold_analysis.csv")

distribution = px.histogram(
    predictions, x="churn_probability", color="Résiliation réelle", barmode="overlay", nbins=25,
    labels={"churn_probability": "Probabilité de résiliation"},
    title="Distribution des scores sur le jeu de test",
)
top_risk = predictions.nlargest(15, "churn_probability")
risk_chart = px.bar(
    top_risk, x="customerID", y="churn_probability", color="Résiliation réelle",
    labels={"customerID": "Client", "churn_probability": "Probabilité de résiliation"},
    title="Clients les plus à risque dans le jeu de test",
)
risk_chart.update_yaxes(tickformat=".0%")


def card(label, value):
    return html.Div([html.P(label), html.Strong(f"{value:.1%}")], className="card")


app = dash.Dash(__name__)
app.title = "ChurnScope"
app.layout = html.Main(className="page", children=[
    html.H1("ChurnScope"),
    html.P("Analyse du risque de résiliation client", className="lead"),
    html.P(f"Modèle retenu : {best_name} (validation croisée sur le jeu d'entraînement)", className="model"),
    html.Div(className="cards", children=[
        card("AUC-ROC · test", metrics["roc_auc"]), card("Rappel · test", metrics["recall"]),
        card("Précision · test", metrics["precision"]), card("AP · test", metrics["average_precision"]),
    ]),
    html.P(f"Référence sans modèle : AUC-ROC {baseline['roc_auc']:.2f} ; "
           f"accuracy {baseline['accuracy']:.1%} en prédisant toujours « reste »."),
    html.H2("Comparaison des modèles"),
    html.P(f"ROC-AUC moyenne sur {report['selection']['folds']} plis du jeu d'entraînement. "
           "Le jeu de test est réservé à l'évaluation finale."),
    html.Table([
        html.Thead(html.Tr([html.Th("Modèle"), html.Th("AUC moyenne"), html.Th("Écart-type")])),
        html.Tbody([
            html.Tr([html.Td(name), html.Td(f"{result['roc_auc_mean']:.3f}"),
                     html.Td(f"{result['roc_auc_std']:.3f}")])
            for name, result in cv_results.items()
        ]),
    ], className="results-table"),
    dcc.Graph(figure=distribution), dcc.Graph(figure=risk_chart),
    html.H2("Où le modèle se trompe"),
    html.P("Erreurs au seuil de 0,5 sur le jeu de test, regroupées par type de contrat. "
           "Ces groupes n'ont pas servi à choisir le modèle."),
    html.Div(html.Table([
        html.Thead(html.Tr([html.Th("Contrat"), html.Th("Clients"), html.Th("Résiliations réelles"),
                              html.Th("Résiliations manquées"), html.Th("Fausses alertes")])),
        html.Tbody([
            html.Tr([html.Td(row.group), html.Td(int(row.clients)), html.Td(int(row.actual_churn)),
                     html.Td(int(row.missed_churn)), html.Td(int(row.false_alerts))])
            for row in contract_errors.itertuples()
        ]),
    ], className="results-table"), className="table-wrap"),
    html.H2("Effet du seuil de décision"),
    html.P("Comparaison descriptive après l'évaluation finale : un seuil plus bas détecte davantage de départs, "
           "mais déclenche aussi plus de fausses alertes. Aucun seuil n'est optimisé sur ce test."),
    html.Div(html.Table([
        html.Thead(html.Tr([html.Th("Seuil"), html.Th("Clients signalés"), html.Th("Précision"),
                              html.Th("Rappel"), html.Th("Fausses alertes"), html.Th("Départs manqués")])),
        html.Tbody([
            html.Tr([html.Td(f"{row.threshold:.1f}"), html.Td(int(row.flagged_clients)),
                     html.Td(f"{row.precision:.1%}"), html.Td(f"{row.recall:.1%}"),
                     html.Td(int(row.false_alerts)), html.Td(int(row.missed_churn))])
            for row in thresholds.itertuples()
        ]),
    ], className="results-table"), className="table-wrap"),
    html.H2("Lecture métier"),
    html.Ul([
        html.Li("Ces graphiques portent uniquement sur les clients du jeu de test, pas sur une utilisation en production."),
        html.Li(f"Au seuil de {report['decision_threshold']:.1f}, le modèle repère "
                f"{metrics['recall']:.1%} des clients qui résilient ; le seuil doit être choisi selon le coût des actions de rétention."),
        html.Li("Les liens observés avec le type de contrat, le service Internet et l'ancienneté ne démontrent pas de causalité."),
        html.Li("La courbe précision-rappel, la matrice de confusion, l'audit des données et l'analyse des variables sont dans reports/."),
    ]),
])

if __name__ == "__main__":
    app.run(debug=True)
