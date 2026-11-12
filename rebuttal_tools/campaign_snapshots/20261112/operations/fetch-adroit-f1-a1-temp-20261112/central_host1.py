"""Archive the authorized server1 expansion, preserving all older experiments."""
from central_collector import *

while True:
    try:
        result = once(1, initial=False)
        if result['passed']:
            base = ROOT / 'hosts/host1/workspace' / NEW
            proof = json.loads((base / 'transfer-proof.json').read_text())
            assert all(digest(base / name) == sha for name, sha in proof.items())
            remote(1, 'archive-ready', dict(passed=True, hashes=proof,
                    central=str(base), verified_unix=time.time()))
    except Exception as e:
        (ROOT / 'host1-archive-error.json').write_text(json.dumps(
            dict(at=time.time(), error=repr(e), traceback=traceback.format_exc())))
    time.sleep(45)
