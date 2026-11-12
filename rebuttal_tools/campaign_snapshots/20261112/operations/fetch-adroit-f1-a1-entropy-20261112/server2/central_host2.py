from central_collector import *
while True:
 try:
  result=once(2,initial=False)
  if result['passed']:
   base=ROOT/'hosts/host2/workspace'/NEW
   proof=json.loads((base/'transfer-proof.json').read_text())
   assert all(digest(base/name)==sha for name,sha in proof.items())
   remote(2,'archive-ready',dict(passed=True,hashes=proof,central=str(base),verified_unix=time.time()))
 except Exception as e:
  (ROOT/'host2-archive-error.json').write_text(json.dumps(dict(at=time.time(),error=repr(e))))
 time.sleep(45)
