import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
while True:
 subprocess.run([sys.executable,'-m','experiments.gmm_mode_gradient.report'],cwd=ROOT,check=False)
 if len(list((ROOT/'runtime/runs').glob('*/COMPLETE.json')))==96:break
 time.sleep(300)
