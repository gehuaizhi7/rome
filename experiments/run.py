import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

def main(
  alg_name: str
  model_name: Union[str, Tuple],
  hparams_fname: str,
  continue_from_run: str,
):
  # Set algorithm-specific variables
  params_class, apply_algo = ALG_DICT[alg_name]

  # Determine run directory
  if continue_from_run is not None:
      run_dir = RESULTS_DIR / dir_name / continue_from_run
      assert (
          run_dir.exists()
      ), f"If continuing from run, {continue_from_run} must exist!"
  else:
      alg_dir = RESULTS_DIR / dir_name
      if alg_dir.exists():
          id_list = [
              int(str(x).split("_")[-1])
              for x in alg_dir.iterdir()
              if str(x).split("_")[-1].isnumeric()
          ]
          run_id = 0 if not id_list else max(id_list) + 1
      else:
          run_id = 0
      run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
      run_dir.mkdir(parents=True, exist_ok=True)
  print(f"Results will be stored at {run_dir}")

  # Get run hyperparameters
  params_path = (
      run_dir / "params.json"
      if continue_from_run is not None
      else HPARAMS_DIR / alg_name / hparams_fname
  )
  hparams = params_class.from_json(params_path)
  if not (run_dir / "params.json").exists():
      shutil.copyfile(params_path, run_dir / "params.json")
  print(f"Executing {alg_name} with parameters {hparams}")

  # Instantiate vanilla model
  print("Instantiating model")
  if type(model_name) is str:
      model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
      tok = AutoTokenizer.from_pretrained(model_name)
      tok.pad_token = tok.eos_token
  else:
      model, tok = model_name








