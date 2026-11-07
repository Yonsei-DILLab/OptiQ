"""Install only this study's supervisor entries; starts are explicit."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--user',default='heejoonorm')
    a=p.parse_args();root=a.root.resolve();source=root/'source'
    (root/'logs').mkdir(parents=True,exist_ok=True)
    entries=[]
    for phase in ['preflight','train']:
        for gpu in range(4):
            name=f'optiq_kl_nm_{phase}_{gpu}'
            entries.append(f'''[program:{name}]
command={a.python} {source}/worker.py --root {root} --phase {phase} --gpu {gpu}
directory={source}
user={a.user}
autostart=false
autorestart=false
startsecs=1
stopwaitsecs=180
stopasgroup=true
killasgroup=true
environment=PYTHONUNBUFFERED="1"
stdout_logfile={root}/logs/{name}.log
redirect_stderr=true
stdout_logfile_maxbytes=20000000
stdout_logfile_backups=2
''')
    entries.append(f'''[program:optiq_kl_nm_monitor]
command={a.python} {source}/monitor.py --root {root}
directory={source}
user={a.user}
autostart=false
autorestart=unexpected
startsecs=1
stopasgroup=true
killasgroup=true
environment=PYTHONUNBUFFERED="1"
stdout_logfile={root}/logs/monitor.log
redirect_stderr=true
stdout_logfile_maxbytes=20000000
stdout_logfile_backups=2
''')
    Path('/etc/supervisor/conf.d/optiq-kl-nm.conf').write_text('\n'.join(entries))


if __name__=='__main__':main()
