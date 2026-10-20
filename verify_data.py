"""Verify the exact benchmark artifacts used by the reported experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED = {
    "metr-la.h5": "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05",
    "adj_mx_METR-LA.pkl": "a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3",
    "pems_node_values.npy": "6e1503b6d76a46dd6f282468d76c14cdda0cb99073ba9edb77897594895b8eb9",
    "pems_adj_mat.npy": "bfe332913480221f0c1e67ff6c7a5ea8896a70a8fc2f6ff04b052842b91072b9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    failures = []
    for name, expected in EXPECTED.items():
        path = data_dir / name
        if not path.exists():
            failures.append(f"missing: {path}")
            continue
        observed = sha256(path)
        state = "OK" if observed == expected else "MISMATCH"
        print(f"{state}  {name}  {observed}")
        if observed != expected:
            failures.append(f"checksum mismatch: {path}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
