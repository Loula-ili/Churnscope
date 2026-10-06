# Résumé des résultats — ChurnScope

## Objectif

Prédire le risque de résiliation d'un client télécom pour aider l'équipe de rétention à prioriser ses actions.

## Données et préparation

- 7 043 clients, 21 colonnes ; cible : `Churn`.
- Les 11 valeurs textuelles vides de `TotalCharges` concernent des clients avec une ancienneté nulle. Elles sont transformées en `0`.
- Aucun identifiant dupliqué, montant négatif, catégorie inattendue ou service incohérent ; 22 profils identiques hors identifiant sont conservés car ils peuvent représenter des clients distincts. L'audit des 21 colonnes est détaillé dans `DATA_AUDIT.md` et `data_audit.json`.
- Les variables numériques sont standardisées et les variables catégorielles encodées avant l'entraînement.

## Méthode et comparaison des modèles

Le jeu de test (20 % des clients, 1 409 lignes) est mis de côté. Les trois modèles sont comparés par validation croisée stratifiée à 5 plis sur l'entraînement. L'encodage et la standardisation sont appris dans chaque pli via un pipeline : ils ne voient pas les données de validation avant la prédiction.

| Modèle | ROC-AUC moyen en validation | Écart-type |
| --- | ---: | ---: |
| Régression logistique | 0,8460 | 0,0124 |
| Random Forest | 0,8417 | 0,0085 |
| Gradient Boosting | 0,8481 | 0,0126 |

Le Gradient Boosting a la moyenne la plus élevée, mais l'écart avec la régression logistique est faible. Ce seul résultat ne démontre pas une supériorité générale.

## Évaluation finale sur le test

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting | 0,8027 | 0,6655 | 0,5160 | 0,5813 | 0,8433 | 0,6600 |
| Référence « reste » | 0,7346 | 0 | 0 | 0 | 0,5000 | 0,2654 |

Le seuil de décision est 0,5. Le modèle laisse donc passer près de la moitié des résiliations réelles ; une équipe métier devrait choisir le seuil selon le coût des offres de rétention et des clients perdus.

La matrice de confusion du test contient 193 vrais positifs, 181 faux négatifs, 97 faux positifs et 938 vrais négatifs. Dans ce découpage, le modèle ne détecte aucun des 45 départs observés parmi les contrats d'un ou deux ans. La comparaison descriptive des seuils montre qu'à 0,3 le rappel atteint 0,751, avec 249 fausses alertes contre 97 à 0,5. Ce constat ne justifie pas de régler le seuil sur le test : il faudrait le faire sur une validation distincte et selon le coût métier.

## Interprétation métier

- Les contrats mensuels présentent un taux de résiliation nettement plus élevé que les contrats d'un ou deux ans.
- Les clients avec fibre optique sont davantage exposés au churn dans ce jeu de données.
- Les clients qui résilient ont généralement une ancienneté plus faible : les premiers mois sont donc une période à surveiller.
- Le paiement par chèque électronique et des montants mensuels plus élevés sont aussi associés au churn dans cet échantillon.
- Ces associations ne démontrent pas que le contrat ou le service Internet cause la résiliation.

## Livrables

- `best_model.joblib` : pipeline entraîné localement, non versionné ;
- `metrics.json` : validation croisée, test final et référence naïve ;
- `feature_importance.csv` et les graphiques : éléments d'interprétation ;
- `DATA_AUDIT.md` et `data_audit.json` : contrôles des 21 colonnes ;
- `error_analysis.csv` et `threshold_analysis.csv` : erreurs par segment et compromis selon le seuil ;
- `test_predictions.csv` : prédictions et scores de risque sur le jeu de test ;
- `app.py` : dashboard Dash local.

## Limites

Un seul jeu de test final, pas d'intervalle de confiance ; hyperparamètres fixés sans recherche systématique ; données publiques et statiques, sans suivi temporel ni démonstration de performance en production. Le dashboard montre les scores du test, pas une interface de prédiction pour de nouveaux clients.
