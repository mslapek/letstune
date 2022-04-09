from typing import Any

import sklearn.model_selection
import xgboost as xgb
from sklearn.datasets import make_classification

import letstune
from letstune import rand

NUM_CLASSES = 3


class BoosterParams(letstune.Params):
    max_depth: int = rand.ints(2, 6)  # type: ignore
    eta: float = rand.uniform(0.01, 0.2, log=True)  # type: ignore

    def create_model(self, **kwargs: Any) -> xgb.Booster:
        return xgb.Booster(  # type: ignore
            {
                **self.to_dict(),
                "objective": "multi:softmax",
                "num_class": NUM_CLASSES,
            },
            **kwargs,
        )


class ClassificationTrainer(letstune.EpochTrainer[BoosterParams]):
    metric = letstune.Metric("valid_accuracy")

    def load_dataset(self, dataset: Any) -> None:
        X, y = make_classification(
            n_samples=2000,
            n_informative=5,
            n_classes=NUM_CLASSES,
            random_state=42,
        )

        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
            X,
            y,
            test_size=0.33,
            random_state=42,
        )

        self.d_train = xgb.DMatrix(data=X_train, label=y_train)
        self.d_valid = xgb.DMatrix(data=X_valid, label=y_valid)

        self.y_valid = y_valid

    def create_model(self, params: BoosterParams, model_file: Any = None) -> None:
        self.model = params.create_model(
            cache=[self.d_train, self.d_valid],
            model_file=model_file,
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.update(
            self.d_train,
            epoch,
        )
        y_valid_pred = self.model.predict(self.d_valid)
        return {
            "valid_accuracy": sklearn.metrics.accuracy_score(
                self.y_valid, y_valid_pred
            ),
        }

    def load(self, checkpoint: Any, params: BoosterParams) -> None:
        old_model = checkpoint.load_xgboost()
        self.create_model(params, old_model)

    def save(self, checkpoint: Any) -> None:
        checkpoint.save_xgboost(self.model)


trainer = ClassificationTrainer()
trainer.load_dataset(None)

params = BoosterParams(max_depth=2, eta=0.5)
trainer.create_model(params)

metrics = [trainer.train_epoch(epoch) for epoch in range(10)]


def test_model_has_all_metrics() -> None:
    for m in metrics:
        assert set(m) == {"valid_accuracy"}


def test_model_has_good_metrics() -> None:
    assert metrics[0]["valid_accuracy"] > 0.5
    assert metrics[-1]["valid_accuracy"] > 0.75
