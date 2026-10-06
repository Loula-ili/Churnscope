"""Contrôles simples des étapes qui peuvent fausser l'analyse."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import train


def sample_clients() -> pd.DataFrame:
    """Deux profils valides, dont un nouveau client sans TotalCharges."""
    return pd.DataFrame({
        "customerID": ["A", "B"], "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1], "Partner": ["No", "Yes"],
        "Dependents": ["No", "No"], "tenure": [0, 12],
        "PhoneService": ["No", "Yes"], "MultipleLines": ["No phone service", "No"],
        "InternetService": ["No", "DSL"],
        "OnlineSecurity": ["No internet service", "Yes"],
        "OnlineBackup": ["No internet service", "No"],
        "DeviceProtection": ["No internet service", "No"],
        "TechSupport": ["No internet service", "Yes"],
        "StreamingTV": ["No internet service", "No"],
        "StreamingMovies": ["No internet service", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [20.0, 30.0], "TotalCharges": [" ", "120.5"],
        "Churn": ["No", "Yes"],
    })


class PreparationTests(unittest.TestCase):
    def test_zero_tenure_explains_empty_total_charges(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "clients.csv"
            sample_clients().to_csv(source, index=False)
            with patch.object(train, "RAW", source):
                X, y, ids, raw, audit = train.load_data()
            self.assertEqual(raw["TotalCharges"].tolist(), [0.0, 120.5])
            self.assertNotIn("customerID", X.columns)
            self.assertNotIn("Churn", X.columns)
            self.assertEqual(y.tolist(), [0, 1])
            self.assertEqual(ids.tolist(), ["A", "B"])
            self.assertEqual(audit["empty_total_charges_filled_with_zero"], 1)
            self.assertEqual(len(audit["columns"]), 21)

    def test_unexplained_missing_charge_is_rejected(self):
        clients = sample_clients()
        clients.loc[0, "tenure"] = 3
        with self.assertRaisesRegex(ValueError, "TotalCharges"):
            train.audit_raw_data(clients)

    def test_duplicate_id_is_rejected(self):
        clients = sample_clients()
        clients.loc[1, "customerID"] = "A"
        with self.assertRaisesRegex(ValueError, "identifiants"):
            train.audit_raw_data(clients)

    def test_unexpected_category_is_rejected(self):
        clients = sample_clients()
        clients.loc[1, "Contract"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Contract"):
            train.audit_raw_data(clients)

    def test_inconsistent_internet_service_is_rejected(self):
        clients = sample_clients()
        clients.loc[0, "OnlineSecurity"] = "Yes"
        with self.assertRaisesRegex(ValueError, "OnlineSecurity"):
            train.audit_raw_data(clients)

    def test_negative_monthly_charge_is_rejected(self):
        clients = sample_clients()
        clients.loc[1, "MonthlyCharges"] = -1
        with self.assertRaisesRegex(ValueError, "MonthlyCharges"):
            train.audit_raw_data(clients)

    def test_text_instead_of_total_charges_is_rejected(self):
        clients = sample_clients()
        clients.loc[0, "TotalCharges"] = "unknown"
        with self.assertRaisesRegex(ValueError, "non numérique"):
            train.audit_raw_data(clients)

    def test_missing_target_class_is_rejected(self):
        clients = sample_clients()
        clients["Churn"] = "No"
        with self.assertRaisesRegex(ValueError, "deux classes"):
            train.audit_raw_data(clients)

    def test_unseen_category_can_be_transformed(self):
        training = pd.DataFrame({"tenure": [0, 5], "Contract": ["Month", "Year"]})
        new_client = pd.DataFrame({"tenure": [2], "Contract": ["Other"]})
        preprocessing = train.make_preprocessor(training).fit(training)
        self.assertEqual(preprocessing.transform(new_client).shape[1],
                         preprocessing.transform(training).shape[1])

    def test_constant_baseline_has_no_false_precision(self):
        result = train.metrics(pd.Series([0, 1, 0, 1]), pd.Series([0.25] * 4))
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["roc_auc"], 0.5)

    def test_error_counts_and_threshold_tradeoff(self):
        scored = pd.DataFrame({
            "Contract": ["Month-to-month"] * 3 + ["One year"],
            "tenure": [2, 4, 6, 30],
            "actual_churn": [1, 1, 0, 0],
            "predicted_churn": [1, 0, 1, 0],
        })
        groups = train.make_error_analysis(scored)
        monthly = groups.loc[(groups["segment"] == "Contrat") &
                             (groups["group"] == "Month-to-month")].iloc[0]
        self.assertEqual(int(monthly["missed_churn"]), 1)
        self.assertEqual(int(monthly["false_alerts"]), 1)
        self.assertEqual(monthly["recall"], 0.5)
        thresholds = train.make_threshold_analysis(
            pd.Series([1, 1, 0, 0]), pd.Series([0.9, 0.4, 0.6, 0.1]).to_numpy()
        )
        middle = thresholds.loc[thresholds["threshold"] == 0.5].iloc[0]
        self.assertEqual(int(middle["missed_churn"]), 1)
        self.assertEqual(int(middle["false_alerts"]), 1)


if __name__ == "__main__":
    unittest.main()
