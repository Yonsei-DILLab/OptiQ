import argparse
import time
from pathlib import Path
from report import report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    while True:
        try:report(a.root)
        except Exception as e:print(repr(e),flush=True)
        time.sleep(60)
