import json
import yaml
import itertools
import sys
from pathlib import Path                                                                               

parent_dir = Path(__file__).resolve().parent                                                           
sys.path.append(str(parent_dir))                          
                                                                                                        
import pandas as pd
from sklearn.model_selection import cross_validate                                                    
                                                        
from config_schema import ExperimentConfig
from pipeline.encoder import get_encoder
from pipeline.feature_set import build_X                                                               
from pipeline.model import get_model
                                                                                                        
                                                                                                        
def run_experiment(cfg: ExperimentConfig, df_raw: pd.DataFrame, custom_bins: dict):                    
    encoder = get_encoder(cfg.encoding)                                                                
    X, y = build_X(df_raw, cfg.binning, encoder, cfg.feature_set, cfg.scaler, custom_bins)
    clf = get_model(cfg.model)                                                                         
    scores = cross_validate(clf, X, y, cv=cfg.cv_folds,                                                    
                          scoring=["roc_auc", "accuracy", "precision", "recall", "f1"])
    
    metrics = {
      f"{name}_mean": round(scores[f"test_{name}"].mean(), 4)                                            
      for name in ["roc_auc", "accuracy", "precision", "recall", "f1"]                                   
    } 
    
    return {                                                                                           
          "run_id": cfg.run_id,                             
          **metrics,
          **{k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
      }                                                   
                                                                                                        
                                                                                                        
if __name__ == "__main__":
    # Load data                                                                                        
    df_raw = pd.read_csv(parent_dir.parent / "dataset" / "train.csv")
                                                                                                        
    # Load config
    with open(parent_dir / "configs" / "grid_v1.yaml") as f:                                           
        config = yaml.safe_load(f)                        
                                                                                                        
    custom_bins = config.get("custom_bins", {})
                                                                                                        
    # Generate all combinations                           
    combos = itertools.product(
        config["encodings"],
        config["binnings"],                                                                            
        config["feature_sets"],
        config["scalers"],                                                                             
        config["models"],                                 
    )

    # Run each experiment
    results_path = parent_dir / "results.jsonl"
    with open(results_path, "a") as out:                                                               
        for enc, bin_, feat, scaler, model in combos:
            cfg = ExperimentConfig(                                                                    
                encoding=enc, binning=bin_, feature_set=feat,
                scaler=scaler, model=model                                                             
            )                                             
            try:
                result = run_experiment(cfg, df_raw, custom_bins)                                      
                out.write(json.dumps(result) + "\n")
                print(f"OK  {cfg.run_id:60s}  auc={result['roc_auc_mean']}  acc={result['accuracy_mean']} prec={result['precision_mean']}  rec={result['recall_mean']}  f1={result['f1_mean']}")                               
            except Exception as e:                                                                     
                print(f"ERR {cfg.run_id}: {e}")