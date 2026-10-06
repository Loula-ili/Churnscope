# Contrôle des données — ChurnScope

- 7 043 clients et 21 colonnes (identifiant et cible inclus).
- 0 identifiant client en double.
- 22 profils identiques hors identifiant : conservés, car des clients distincts peuvent avoir les mêmes caractéristiques.
- 11 valeurs vides de `TotalCharges` pour des clients avec `tenure = 0` : remplacées par 0.
- Aucune autre valeur manquante, catégorie inattendue, valeur numérique négative ou incohérence entre services téléphoniques/Internet.

## Vérification par colonne

| Colonne | Type brut | Valeurs distinctes | Nulles | Chaînes vides |
| --- | --- | ---: | ---: | ---: |
| `customerID` | str | 7043 | 0 | 0 |
| `gender` | str | 2 | 0 | 0 |
| `SeniorCitizen` | int64 | 2 | 0 | 0 |
| `Partner` | str | 2 | 0 | 0 |
| `Dependents` | str | 2 | 0 | 0 |
| `tenure` | int64 | 73 | 0 | 0 |
| `PhoneService` | str | 2 | 0 | 0 |
| `MultipleLines` | str | 3 | 0 | 0 |
| `InternetService` | str | 3 | 0 | 0 |
| `OnlineSecurity` | str | 3 | 0 | 0 |
| `OnlineBackup` | str | 3 | 0 | 0 |
| `DeviceProtection` | str | 3 | 0 | 0 |
| `TechSupport` | str | 3 | 0 | 0 |
| `StreamingTV` | str | 3 | 0 | 0 |
| `StreamingMovies` | str | 3 | 0 | 0 |
| `Contract` | str | 3 | 0 | 0 |
| `PaperlessBilling` | str | 2 | 0 | 0 |
| `PaymentMethod` | str | 4 | 0 | 0 |
| `MonthlyCharges` | float64 | 1585 | 0 | 0 |
| `TotalCharges` | str | 6531 | 0 | 11 |
| `Churn` | str | 2 | 0 | 0 |

Les modalités attendues des variables catégorielles et leurs effectifs sont enregistrés dans `data_audit.json`.
Ces contrôles valident ce jeu de données ; ils ne remplacent pas une revue métier sur des données nouvelles.
