"""
Code-level dependency graph for src/.

Section 6 of the dashboard traces *data* lineage (which file produced which CSV).
This does the other half: which modules import which, and which scripts nothing
imports. Emits mermaid so provenance_tab.py can render it inline.

    python src/audit/import_graph.py           # writes outputs/import_graph.mmd, prints summary
    python src/audit/import_graph.py --test    # self-check
"""
import ast
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC)


def _modules():
    """Local module name -> path, for every .py under src/ (one level of subdirs)."""
    mods = {}
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, SRC)
            name = rel[:-3].replace(os.sep, ".")
            mods[name] = path
            # scripts import siblings bare (`import metrics`) and packages by
            # leaf too (`from regimes.human_actual import ...` -> both forms seen)
            mods.setdefault(os.path.basename(f)[:-3], path)
    return mods


def _imports(path, local):
    """Local module names imported by one file. Unparseable file -> empty set."""
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            # level>0 is a relative import; node.module is None for `from . import x`
            names = [node.module] if node.module else [a.name for a in node.names]
        else:
            continue
        for n in names:
            if not n:
                continue
            for cand in (n, n.split(".")[0], n.split(".")[-1]):
                if cand in local:
                    out.add(_canon(cand, local))
                    break
    return out


def _canon(name, local):
    """Collapse aliases (regimes.human_score / human_score) onto one node id."""
    path = local[name]
    return os.path.relpath(path, SRC)[:-3].replace(os.sep, ".")


def graph():
    """{module: set(modules it imports)} over src/, self-edges dropped."""
    local = _modules()
    canon = {_canon(n, local) for n in local}
    g = {}
    for name in sorted(canon):
        path = local[name] if name in local else local[name.split(".")[-1]]
        g[name] = {d for d in _imports(path, local) if d != name}
    return g


def orphans(g):
    """Modules nothing imports — entry points, or dead code. Can't tell which."""
    imported = {d for deps in g.values() for d in deps}
    return sorted(set(g) - imported)


def _nid(m):
    return m.replace(".", "_")


def mermaid(g=None, hide_orphans=True):
    """Mermaid flowchart of the import edges. Isolated nodes are noise; drop them."""
    g = g or graph()
    edges = [(a, b) for a, deps in g.items() for b in sorted(deps)]
    linked = {m for e in edges for m in e}
    lines = ["flowchart LR"]
    for m in sorted(g):
        if hide_orphans and m not in linked:
            continue
        lines.append(f'  {_nid(m)}["{m}"]')
    for a, b in edges:
        lines.append(f"  {_nid(a)} --> {_nid(b)}")
    # shared modules (2+ importers) are the real coupling; mark them
    fan = {}
    for _, b in edges:
        fan[b] = fan.get(b, 0) + 1
    for m, n in fan.items():
        if n >= 2:
            lines.append(f"  style {_nid(m)} fill:#DBEAFE,stroke:#2563EB")
    return "\n".join(lines)


def _test():
    g = graph()
    assert "dashboard" in g, "dashboard.py not found"
    assert "metrics" in g["dashboard"], f"dashboard should import metrics, got {g['dashboard']}"
    assert "import_graph" in g["provenance_tab"]
    assert "dashboard" not in g["provenance_tab"], "provenance_tab must not import dashboard"
    for m, deps in g.items():
        assert m not in deps, f"self-edge on {m}"
        for d in deps:
            assert d in g, f"{m} -> {d} which is not a known module"
    m = mermaid(g)
    assert m.startswith("flowchart LR")
    assert "-->" in m
    assert "dashboard" in orphans(g), "the entry point should have no importers"
    print(f"ok — {len(g)} modules, {sum(len(v) for v in g.values())} edges, "
          f"{len(orphans(g))} orphans")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    else:
        g = graph()
        os.makedirs(os.path.join(REPO, "outputs"), exist_ok=True)
        out = os.path.join(REPO, "outputs", "import_graph.mmd")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(mermaid(g) + "\n")
        print(f"{len(g)} modules, {sum(len(v) for v in g.values())} import edges -> {out}")
        print("\nnothing imports these (entry points or dead):")
        for m in orphans(g):
            print(f"  {m}")
