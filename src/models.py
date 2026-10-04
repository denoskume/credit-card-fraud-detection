import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from xgboost import XGBClassifier

RANDOM_STATE = 42


def build_logistic_regression(random_state: int = RANDOM_STATE) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )


def build_random_forest(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(random_state: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )


class FraudMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, transaction_features: torch.Tensor) -> torch.Tensor:
        return self.network(transaction_features).squeeze(-1)
