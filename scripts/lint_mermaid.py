"""Cheap structural check on the mermaid blocks: no renderer here, so verify the
things that actually break one -- a missing diagram type, unbalanced quotes or
brackets in a node label, and an edge naming a node id that was never defined."""

import re
import sys
from pathlib import Path

TYPES = ("flowchart", "graph", "stateDiagram", "sequenceDiagram", "classDiagram")
ROOT = Path(__file__).resolve().parents[1]
documents = sys.argv[1:] or sorted(
    str(p) for p in ROOT.glob("**/*.md") if "third_party" not in p.parts and "node_modules" not in p.parts
)
ok = True
for doc in documents:
    text = Path(doc).read_text()
    for n, block in enumerate(re.findall(r"```mermaid\n(.*?)```", text, re.S), 1):
        head = block.strip().splitlines()[0]
        if not head.startswith(TYPES):
            print(f"{doc} #{n}: bad diagram type {head!r}")
            ok = False
        for line in block.splitlines():
            if line.count('"') % 2:
                print(f"{doc} #{n}: odd quote count: {line.strip()[:70]}")
                ok = False
            if line.count("[") != line.count("]") or line.count("{") != line.count("}"):
                print(f"{doc} #{n}: unbalanced bracket: {line.strip()[:70]}")
                ok = False
        if head.startswith(("flowchart", "graph")):
            # A node may be declared anywhere, including as an edge target on the
            # same line: `A[观察] --> B[预抓取]` defines both.
            defined = set(re.findall(r"([A-Za-z][\w]*)\s*[\[({]", block))
            defined |= set(re.findall(r"subgraph\s+([A-Za-z][\w]*)", block))
            used = set()
            for edge in re.findall(r"^\s*([A-Za-z][\w]*)\s*(?:-{2,3}|-\.-|==)", block, re.M):
                used.add(edge)
            for edge in re.findall(r"(?:-->|-\.->|---)\s*(?:\|[^|]*\|\s*)?([A-Za-z][\w]*)", block):
                used.add(edge)
            missing = {u for u in used - defined if u not in {"o", "x"}}
            if missing:
                print(f"{doc} #{n}: edge references undefined node(s): {sorted(missing)}")
                ok = False
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
