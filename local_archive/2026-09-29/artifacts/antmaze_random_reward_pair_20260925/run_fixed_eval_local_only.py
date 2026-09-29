"""Run the frozen fixed-origin evaluator while suppressing W&B writes."""
from pathlib import Path
import runpy
import sys
import wandb


class LocalRun:
    url = None
    def log(self, *args, **kwargs):
        pass
    def define_metric(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass


original_init = wandb.init
def local_evaluation_only(**kwargs):
    if kwargs.get('job_type') == 'evaluation':
        return LocalRun()
    return original_init(**kwargs)

wandb.init = local_evaluation_only

target = Path(sys.argv[1]).resolve()
arguments = sys.argv[2:]
output = Path(arguments[arguments.index('--output') + 1])
sys.argv = [str(target), *arguments]
runpy.run_path(str(target), run_name='__main__')
(output / 'wandb.json').unlink(missing_ok=True)
(output / 'wandb-disabled.json').write_text(
    '{"uploaded":false,"reason":"local fixed-start evaluation only"}\n')
