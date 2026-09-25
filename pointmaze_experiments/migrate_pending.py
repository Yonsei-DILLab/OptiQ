"""Run on vast2: reserve four not-yet-started outputs so frozen queues skip them."""
import json
from pathlib import Path

for maze in ('medium', 'hard'):
    for temperature in (3, 5):
        p = Path(f'/workspace/ibolt-drac-{maze}-T{temperature}-s0-20260926')
        p.mkdir(exist_ok=False)
        (p/'MIGRATED.json').write_text(json.dumps(dict(
            status='migrated_before_start',destination_host='vast4',
            destination_path=str(p),reason='User requests concurrent execution',
            training_started_here=False),indent=2))
        print(p)
