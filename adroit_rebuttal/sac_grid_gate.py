"""Hold the existing baseline continuation until SAC grid confirmation completes."""
import argparse
import json
from pathlib import Path
import time
from .run import atomic


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state', type=Path, required=True)
    p.add_argument('--queue-state', type=Path, required=True); a = p.parse_args()
    while True:
        state = json.loads(a.state.read_text()) if a.state.exists() else {}
        if state.get('status') == 'completed' and state.get('confirmation_runs') == 20:
            return
        atomic(a.queue_state, dict(status='waiting', observed_unix=time.time(),
            reason='Waiting for SAC alpha search and independent confirmation',
            sac_grid_status=state.get('status', 'pending'), sac_grid_phase=state.get('phase')))
        time.sleep(30)


if __name__ == '__main__':main()
