"""Register an already committed campaign on the existing experiment supervisor."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument("--name",required=True)
    p.add_argument("--smoke",action="store_true");p.add_argument("--methods",nargs="+")
    a=p.parse_args()
    if not all(c.isalnum() or c=="-" for c in a.name):raise ValueError("Invalid name")
    source=Path(__file__).resolve().parents[1]
    root=Path("/home/heechan/optiq-experiments")/a.name
    root.mkdir(exist_ok=True)
    conf=Path("/home/heechan/OptiQ-ops/supervisor/jobs")/(a.name+".conf")
    if conf.exists() or (root/"manifest.json").exists():raise RuntimeError("Already registered")
    command=f"{sys.executable} -m antmaze.controller --root {root}"
    if a.smoke:command+=" --smoke"
    if a.methods:
        assert a.smoke and all(x in ("optiq","sac","dipo","meow","mfpo","sql") for x in a.methods)
        command+=" --methods "+" ".join(a.methods)
    conf.write_text(f"[program:{a.name}]\ncommand={command}\ndirectory={source}\n"
        f"autostart=false\nautorestart=false\nstartsecs=1\nstopasgroup=false\nkillasgroup=false\n"
        f"redirect_stderr=true\nstdout_logfile={root}/controller.log\nstdout_logfile_maxbytes=0\n"
        'environment=PYTHONUNBUFFERED="1",MPLBACKEND="Agg"\n')
    ctl=["supervisorctl","-c","/home/heechan/OptiQ-ops/supervisor/supervisord.conf"]
    for cmd in (["reread"],["update",a.name],["start",a.name]):subprocess.run(ctl+cmd,check=True)


if __name__=="__main__":main()
