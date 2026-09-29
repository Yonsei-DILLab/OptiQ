"""Collect one immutable policy archive at a time to bound download size."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
from collect_results import collect, HOSTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=HOSTS, required=True)
    parser.add_argument('--job', action='append')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'results' / args.host / 'manifest.json').read_text())
    registered = [job['id'] for job in manifest['jobs']]
    jobs = args.job or registered
    assert all(job in registered for job in jobs)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    results = []
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = {pool.submit(collect, args.host, stamp, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(host=args.host, job=futures[future], error_type=type(exc).__name__)
            results.append(result)
            print(json.dumps(result), flush=True)
    (root / ('latest-chunked-' + args.host + '.json')).write_text(json.dumps(
        dict(time_utc=stamp, jobs=results), indent=2) + '\n')
    return int(any('error_type' in item for item in results))

if __name__ == '__main__':
    raise SystemExit(main())
