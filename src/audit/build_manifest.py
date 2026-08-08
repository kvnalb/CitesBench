"""
Generate MANIFEST.md — what each script is, and which script produced each output.

Provenance is derived, not maintained by hand: every "outputs/..." and "data/..."
string literal in every script is extracted from the source, so the map cannot drift
from the code. What it cannot see is a path built at runtime (f-strings with a
variable, e.g. per-rubric or per-model suffixes); those are reported separately as
patterns rather than silently missed.

Three sections, and the last two are the ones that answer "where did this come from":
  Scripts   one line per script, its group, and its first docstring line
  Outputs   every file in outputs/ and data/ with its producing script, or ORPHAN
  Unclaimed patterns  runtime-built paths, and scripts that write nothing

Run: python src/audit/build_manifest.py
"""
import os
import re
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "MANIFEST.md")
GROUPS = {
    "fetch": "pulls from an external API or dump — incremental and resumable",
    "build": "turns fetched data into analysis tables and frozen samples",
    "probes": "sends prompts to a model (leakage / recall / review probes)",
    "analysis": "reads tables, computes results, writes reports and figures",
    "app": "the Streamlit dashboard and its pages",
    "audit": "checks the repo itself: data quality, prompts, provenance",
    ".": "shared modules imported by the above",
}
# a literal path, or one with an interpolated segment we keep as a pattern
LIT = re.compile(r'["\'](?:outputs|data)/[^"\'\s]+["\']')
# CONST = "outputs/x.csv" — scripts name the path once and write through the name
ASSIGN = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*["\']((?:outputs|data)/[^"\']+)["\']',
                    re.M)


def write_sites(src, token):
    """True if `token` (a literal with quotes, or a constant name) is written through.

    Without this, a script that only reads a file is credited as its producer — the
    dashboard reads two dozen tables it does not make.
    """
    t = re.escape(token)
    pats = [rf'to_csv\(\s*{t}', rf'to_json\(\s*{t}', rf'savefig\(\s*{t}',
            rf'open\(\s*{t}\s*,\s*["\'][wa]', rf'to_parquet\(\s*{t}',
            rf'write_text\(\s*{t}', rf'\.to_pickle\(\s*{t}',
            # the path is handed to a local writer function, e.g.
            # write_markdown("outputs/data_audit.md") in data_audit.py
            rf'\b(?:write|save|export|dump|render|emit)\w*\(\s*{t}']
    return any(re.search(p, src) for p in pats)


def classify(src):
    """Returns (written, read, patterns) sets of paths for one script."""
    names = dict((m.group(1), m.group(2)) for m in ASSIGN.finditer(src))
    written, patterns = set(), set()
    for name, path in names.items():
        if write_sites(src, name):
            (patterns if "{" in path else written).add(path)
    for lit in LIT.findall(src):
        path = lit[1:-1]
        if "{" in path or "%" in path:
            patterns.add(path)
        elif write_sites(src, lit):
            written.add(path)
    all_lit = {l[1:-1] for l in LIT.findall(src)} | set(names.values())
    read = {p for p in all_lit if p not in written and p not in patterns}
    return written, read, patterns


def rel(p):
    return os.path.relpath(p, ROOT)


def scripts():
    out = []
    for g in GROUPS:
        d = os.path.join(ROOT, "src", g) if g != "." else os.path.join(ROOT, "src")
        for p in sorted(glob.glob(os.path.join(d, "*.py"))):
            if g == "." and os.path.dirname(rel(p)) != "src":
                continue
            src = open(p, encoding="utf-8", errors="replace").read()
            m = re.match(r'\s*"""\s*(.+)', src)
            doc = (m.group(1).strip() if m else "").rstrip(".")
            out.append({"path": rel(p), "group": g, "doc": doc, "src": src})
    # the streamlit pages dir
    for p in sorted(glob.glob(os.path.join(ROOT, "src", "app", "pages", "*.py"))):
        src = open(p, encoding="utf-8", errors="replace").read()
        m = re.match(r'\s*"""\s*(.+)', src)
        out.append({"path": rel(p), "group": "app",
                    "doc": (m.group(1).strip() if m else "").rstrip("."), "src": src})
    return out


def main():
    ss = scripts()

    writes, reads, patterns = {}, {}, {}
    for s in ss:
        w, r, pat = classify(s["src"])
        for p in w:
            writes.setdefault(p, []).append(s["path"])
        for p in r:
            reads.setdefault(p, []).append(s["path"])
        for p in pat:
            patterns.setdefault(p, []).append(s["path"])

    # Cache directories hold thousands of API-response files whose provenance is the
    # directory, not the file. Roll each up to one row so real outputs stay legible.
    ROLLUP_MIN = 20
    all_files = [rel(p) for d in ("outputs", "data")
                 for p in glob.glob(os.path.join(ROOT, d, "**", "*"), recursive=True)
                 if os.path.isfile(p)]
    bydir = {}
    for p in all_files:
        bydir.setdefault(os.path.dirname(p), []).append(p)
    present, rolled = [], {}  # noqa: E501  (rolled filled below)
    for d, files in bydir.items():
        # never roll up outputs/ or data/ themselves — only nested cache dirs
        if d not in ("outputs", "data") and len(files) >= ROLLUP_MIN:
            rolled[d] = len(files)
        else:
            present += files
    present = sorted(present)

    L = ["# MANIFEST", "",
         "Generated by `python src/audit/build_manifest.py` — do not edit. Producers "
         "are extracted from the string literals in each script, so this file cannot "
         "drift from the code.", "",
         f"{len(ss)} scripts, {len(present)} files under `outputs/` and `data/`.", "",
         "## Layout", "", "```",
         "src/fetch/      pulls from an external API or dump (incremental, resumable)",
         "src/build/      builds analysis tables and frozen samples",
         "src/probes/     sends prompts to a model",
         "src/analysis/   computes results, writes reports and figures",
         "src/app/        Streamlit dashboard + pages/",
         "src/audit/      audits the repo: data quality, prompts, this manifest",
         "src/*.py        shared modules (prompts, metrics, baselines) + regimes/",
         "prompts/        every prompt as a .txt template",
         "data/           read-only inputs; only fetch scripts may write here",
         "outputs/        everything generated",
         "Archive/        historical, never imported or run",
         "```", "",
         "All scripts run from the repo root: `python src/<group>/<name>.py`.", ""]

    L += ["## Scripts", ""]
    for g, blurb in GROUPS.items():
        rows = [s for s in ss if s["group"] == g]
        if not rows:
            continue
        L += [f"### `src/{g}/`" if g != "." else "### `src/` (shared)", "",
              f"_{blurb}_", "", "| script | what it does |", "|---|---|"]
        for s in rows:
            L.append(f"| `{s['path']}` | {s['doc'][:120]} |")
        L.append("")

    L += ["## Outputs and their producers", "",
          "`consumed by` lists scripts that read the file — that is the blast radius "
          "if it changes.", "",
          "| file | produced by | consumed by |", "|---|---|---|"]
    # A pattern like "outputs/eval_table_{year}.csv" does name a real file — resolve it
    # by regex so those files get their producer instead of reading as orphans.
    pat_re = []
    for pat, owners in patterns.items():
        rx = re.escape(pat)
        rx = re.sub(r"\\\{[^}]*\\\}", "[^/]+", rx).replace(r"\%s", "[^/]+")
        pat_re.append((re.compile(f"^{rx}$"), pat, owners))

    def producers(p):
        if writes.get(p):
            return sorted(set(writes[p])), "literal"
        for rx, pat, owners in pat_re:
            if rx.match(p):
                return sorted(set(owners)), f"pattern `{pat}`"
        return [], ""

    orphans, shell_logs = [], []
    for p in present:
        prod, how = producers(p)
        cons = sorted(set(reads.get(p, [])))
        if prod:
            cell = ", ".join(f"`{x}`" for x in prod)
            if how.startswith("pattern"):
                cell += f" (via {how})"
        elif p.endswith(".log"):
            # logs come from shell redirection (`python src/... > outputs/x.log`), which
            # is invisible to source scanning. Guess the owner by name overlap.
            stem = os.path.basename(p)[:-4]
            guess = [s["path"] for s in ss
                     if os.path.basename(s["path"])[:-3] in stem
                     or stem.startswith(os.path.basename(s["path"])[:-3][:12])]
            shell_logs.append((p, guess))
            cell = ("shell redirect, likely " + ", ".join(f"`{g}`" for g in sorted(set(guess))[:2])
                    if guess else "**shell redirect — owner unknown**")
        else:
            cell = ("**ORPHAN — read but never written here**" if cons
                    else "**ORPHAN — no script names this path**")
            orphans.append(p)
        L.append(f"| `{p}` | {cell} | {', '.join(f'`{x}`' for x in cons) or '—'} |")
    for d, n in sorted(rolled.items()):
        prod = sorted({s for pat, v in patterns.items() if pat.startswith(d) for s in v})
        L.append(f"| `{d}/` ({n:,} files, rolled up) | "
                 f"{', '.join(f'`{x}`' for x in prod) or '**ORPHAN**'} | — |")
    L.append("")

    multi = {p: sorted(set(v)) for p, v in writes.items() if len(set(v)) > 1}
    if multi:
        L += ["### Files written by more than one script", "",
              "These are mutated in place — one script creates the file, others add "
              "columns to it. Run order matters, and a rerun of the creator silently "
              "drops the later columns.", "",
              "| file | writers |", "|---|---|"]
        for p, v in sorted(multi.items()):
            L.append(f"| `{p}` | {', '.join(f'`{x}`' for x in v)} |")
        L.append("")

    declared_missing = sorted(set(writes) - set(present))
    silent = [s["path"] for s in ss
              if not any(s["path"] in v for v in writes.values())
              and not any(s["path"] in v for v in patterns.values())]
    silent = [p for p in silent if not p.startswith("src/app/")]   # the app only reads

    L += ["## Gaps", "",
          f"**{len(shell_logs)} run logs** — produced by shell redirection "
          f"(`python src/... > outputs/x.log`), which no source scan can see. Owner is "
          f"inferred from the filename and is a guess, not provenance.", "",
          f"**{len(orphans)} files no script in this repo writes.** Split by whether "
          f"anything reads them — the read ones are external inputs (fetched before "
          f"this repo existed, or handed over), the unread ones have no traceable "
          f"provenance at all and are the ones to worry about:", ""]
    inputs = [p for p in orphans if reads.get(p)]
    unknown = [p for p in orphans if not reads.get(p)]
    L += ["", f"_External inputs, read by {len(inputs)} paths:_", ""]
    L += [f"- `{p}` — read by {', '.join(f'`{x}`' for x in sorted(set(reads[p]))[:3])}"
          for p in inputs] or ["- none"]
    L += ["", f"_No producer and no consumer ({len(unknown)}) — dead files, or "
          f"provenance lost:_", ""]
    L += [f"- `{p}`" for p in unknown] or ["- none"]
    L += ["", f"**{len(patterns)} runtime-built paths** — the producer is known but the "
          f"filename depends on a flag or model name, so it cannot be matched to a "
          f"file by string comparison:", ""]
    L += [f"- `{p}` — `{', '.join(sorted(set(v)))}`" for p, v in sorted(patterns.items())] \
        or ["- none"]
    L += ["", f"**{len(declared_missing)} declared but absent** — a script names this "
          f"path but the file is not on disk (never run, or cleaned up):", ""]
    L += [f"- `{p}`" for p in declared_missing] or ["- none"]
    L += ["", f"**{len(silent)} scripts write nothing** — libraries, or they only print:",
          ""]
    L += [f"- `{p}`" for p in silent] or ["- none"]

    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"Wrote {rel(OUT)}")
    print(f"  {len(ss)} scripts, {len(present)} files, {len(orphans)} orphans, "
          f"{len(patterns)} runtime patterns, {len(declared_missing)} declared-absent, "
          f"{len(silent)} write nothing")

    print("\n=== selfcheck ===")
    bad = []
    if not present:
        bad.append("no output files found — wrong working directory?")
    # a script that both writes and is the sole reader of a path is fine; a path with
    # several writers is reported in the document, not treated as a failure
    for b in bad:
        print(f"  FAIL {b}")
    if not bad:
        print(f"  {len(present) - len(orphans)}/{len(present)} files attributable to a "
              f"producing script; {len(multi)} written by more than one script")


if __name__ == "__main__":
    main()
