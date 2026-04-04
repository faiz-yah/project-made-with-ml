from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    encoding: str
    binning: str
    feature_set: str
    scaler: str
    model: str
    cv_folds: int = 5
    metric: str = "roc_auc"

    @property
    def run_id(self) -> str:
        return f"{self.encoding}_{self.binning}_{self.feature_set}_{self.scaler}_{self.model}"