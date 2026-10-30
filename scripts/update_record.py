#!/usr/bin/env python3
"""Regenerate plugin/minicpm5_mk-0.1.0.dist-info/RECORD (wheel RECORD format) from the files on disk."""
import base64
import hashlib
import os

PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugin")
DIST = "minicpm5_mk-0.1.0.dist-info"


def entry(rel):
    with open(os.path.join(PLUGIN, rel), "rb") as f:
        data = f.read()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{rel},sha256={digest},{len(data)}"


def main():
    rels = []
    for top in ("minicpm5_mk", DIST):
        for dp, dns, fns in os.walk(os.path.join(PLUGIN, top)):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for fn in sorted(fns):
                rel = os.path.relpath(os.path.join(dp, fn), PLUGIN)
                if rel != f"{DIST}/RECORD" and not fn.endswith(".pyc"):
                    rels.append(rel)
    lines = [entry(r) for r in rels] + [f"{DIST}/RECORD,,"]
    with open(os.path.join(PLUGIN, DIST, "RECORD"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {DIST}/RECORD ({len(rels)} files)")


if __name__ == "__main__":
    main()
