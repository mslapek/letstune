import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(kw_only=True, frozen=True)
class LocalCheckpoint:
    path: Path

    def load_pickle(self) -> Any:
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def save_pickle(self, model: Any) -> Any:
        self._remove_if_exists()
        self._make_parent_dir()

        with open(self.path, "xb") as f:
            return pickle.dump(model, f)

    def load_keras(self) -> Any:
        from tensorflow import keras

        return keras.models.load_model(self.path)

    def save_keras(self, model: Any) -> Any:
        self._remove_if_exists()
        self._make_parent_dir()
        model.save(self.path)

    def load_xgboost(self, model_factory: Callable[[Path], Any] | None = None) -> Any:
        p = self.path / "model.json"

        if model_factory is None:
            import xgboost as xgb

            return xgb.Booster(model_file=p)  # type: ignore
        else:
            return model_factory(p)

    def save_xgboost(self, model: Any) -> Any:
        self._remove_if_exists()
        self.path.mkdir(parents=True)
        model.save_model(self.path / "model.json")

    def _remove_if_exists(self) -> None:
        if self.path.exists():
            if self.path.is_file():
                self.path.unlink()
            else:
                shutil.rmtree(self.path)

    def _make_parent_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)


class CheckpointFactory:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def get_checkpoint(
        self, training_id: int, epoch_id: int | None = None
    ) -> LocalCheckpoint:
        path = self._directory / f"training{training_id}"

        if epoch_id is not None:
            path = path / f"epoch{epoch_id}"

        return LocalCheckpoint(path=path)
