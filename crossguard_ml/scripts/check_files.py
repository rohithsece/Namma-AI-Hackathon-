"""Verify all packaged files against manifest.json, without extra packages."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise SystemExit(f'FAIL: {name}')
print(f"PASS: {len(manifest['sha256'])} files verified")
