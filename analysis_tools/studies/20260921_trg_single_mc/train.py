from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'repo'))
import run_optiq_dime as runner
from hydra import initialize_config_dir,compose

def compose_config(overrides=()):
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'repo/configs')):
        return compose(config_name='mujoco_trg_single_mc',overrides=list(overrides))
