"""study2.py - download (optional) + localization study. One self-contained file.

    python study2.py                 # all tasks (downloads graphs if missing)
    python study2.py --limit 30      # first 30 tasks only
    python study2.py --skip-download # use whatever is already in kaggle_data/
Outputs: loc_labels.csv, loc_results.csv, loc_results_by_repo.csv
"""
import csv, glob, json, math, os, re, shutil, sys, zipfile
from collections import Counter, defaultdict
from urllib.parse import quote
print("=== study2.py  VERSION 5 (patch parser + normalized matching) ===")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_ROOT") or os.path.join(HERE, "kaggle_data")
COMP = "gemma-4-developer-agent"
ARGV = sys.argv[1:]
LIMIT = int(ARGV[ARGV.index("--limit") + 1]) if "--limit" in ARGV else None
SKIP = "--skip-download" in ARGV
WITH_EMB = "--emb" in ARGV
for d in ("", "graphs", "embeddings"):
    os.makedirs(os.path.join(DATA, d), exist_ok=True)

# ============================ download ============================
def short(repo): return repo.split("/")[-1].lower()

if not SKIP:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    names, token = [], None
    while True:
        kw = {"page_token": token} if token else {}
        try: res = api.competition_list_files(COMP, page_size=200, **kw)
        except TypeError: res = api.competition_list_files(COMP, **kw)
        files = getattr(res, "files", None) or []
        names += [getattr(f, "name", None) or str(f) for f in files]
        token = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if not token or not files: break
    by_base = {}
    for n in names: by_base.setdefault(os.path.basename(n), n)
    print(f"listed {len(names)} competition files")

    def fetch(remote, local_dir):
        target = os.path.join(local_dir, os.path.basename(remote))
        if os.path.exists(target) and os.path.getsize(target) > 0: return True
        last = None
        for v in (remote, quote(remote, safe="")):
            try: api.competition_download_file(COMP, v, path=local_dir, quiet=True); break
            except Exception as e: last = e
        else:
            print("  FAILED", remote, str(last)[:120]); return False
        z = target + ".zip"
        if os.path.exists(z):
            with zipfile.ZipFile(z) as zf: zf.extractall(local_dir)
            os.remove(z)
        return os.path.exists(target)

    tp = os.path.join(DATA, "tasks.jsonl")
    if os.path.exists(tp) and sum(1 for _ in open(tp)) < 100:   # stale truncated copy from older script
        os.remove(tp)
    assert fetch("tasks.jsonl", DATA), "cannot download tasks.jsonl"

TASKS = [json.loads(l) for l in open(os.path.join(DATA, "tasks.jsonl")) if l.strip()]
if LIMIT: TASKS = TASKS[:LIMIT]
print(len(TASKS), "tasks")

if not SKIP:
    def get_for_task(t, ext, folder):
        iid, commit = t["instance_id"], t.get("base_commit", "")
        ld = os.path.join(DATA, folder); dst = os.path.join(ld, f"{iid}.{ext}")
        if os.path.exists(dst) and os.path.getsize(dst) > 0: return True
        for base in (f"{short(t['repo'])}_{commit}.{ext}", f"{iid}.{ext}"):
            r = by_base.get(base)
            if r and fetch(r, ld):
                src = os.path.join(ld, base)
                if src != dst: shutil.copyfile(src, dst)
                return True
        return False
    ok = 0
    for i, t in enumerate(TASKS, 1):
        g = get_for_task(t, "json", "graphs")
        if WITH_EMB: get_for_task(t, "npz", "embeddings")
        ok += bool(g); print(f"[{i}/{len(TASKS)}] {t['instance_id']} {'ok' if g else 'MISSING'}")
    print(f"graphs downloaded: {ok}/{len(TASKS)}")

# ============================ helpers ============================
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
def words(s): return [w.lower() for w in WORD.findall(s)]
def split_id(w):
    p = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).replace("_", " ").split()
    return [x.lower() for x in p if len(x) > 1] + [w.lower()]
def tokenize(text):
    out = []
    for t in re.findall(r"[A-Za-z][A-Za-z0-9]+", text): out.extend(split_id(t))
    return out

def load_graph(iid):
    p = os.path.join(DATA, "graphs", f"{iid}.json")
    if not os.path.exists(p): return None
    g = json.load(open(p))
    nodes = {n["id"]: (n.get("text") or "") for n in g["nodes"]}
    o, i = defaultdict(set), defaultdict(set)
    for e in g["edges"]:
        s, t = e["source"], e["target"]
        if s in nodes and t in nodes and s != t: o[s].add(t); i[t].add(s)
    return nodes, o, i

def load_emb(iid, nodes):
    import numpy as np
    p = os.path.join(DATA, "embeddings", f"{iid}.npz")
    if not os.path.exists(p): return None
    z = np.load(p); ids, vs = [], []
    for k in z.files:
        nid = k[:-4] if k.endswith(".npy") else k
        if nid in nodes: ids.append(nid); vs.append(z[k].astype("float32"))
    if not ids: return None
    M = np.stack(vs); M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return ids, M

# ============================ gold labels ============================
def parse_patch(patch):
    """Plain unified diffs ('--- a/x' / '+++ b/x') and git diffs. -> {path: [hunk,...]}"""
    files, cur, hunk = {}, None, None
    L = patch.splitlines()
    i = 0
    while i < len(L):
        line = L[i]
        if line.startswith("--- ") and i + 1 < len(L) and L[i + 1].startswith("+++ "):
            old = re.sub(r"^a/", "", line[4:].split("\t")[0].strip())
            new = re.sub(r"^b/", "", L[i + 1][4:].split("\t")[0].strip())
            cur = old if new == "/dev/null" else new
            is_new = (old == "/dev/null")
            files.setdefault(cur, []); hunk = None
            if is_new: files[cur] = "NEWFILE"
            i += 2; continue
        if line.startswith("@@") and cur and files.get(cur) != "NEWFILE":
            m = re.match(r"@@ .*? @@ ?(.*)", line)
            hunk = {"removed": [], "context": [], "added": [], "header": (m.group(1) if m else "")}
            files[cur].append(hunk)
        elif hunk is not None:
            if line.startswith("-"): hunk["removed"].append(line[1:])
            elif line.startswith("+"): hunk["added"].append(line[1:])
            elif line.startswith(" "): hunk["context"].append(line[1:])
        i += 1
    return files

def module_prefixes(path, nodes):
    p = path[:-3] if path.endswith(".py") else path
    p = re.sub(r"/__init__$", "", p)
    parts = p.split("/")
    for k in range(len(parts)):                      # strip leading dirs (src/, lib/, ...) until ids match
        m = ".".join(parts[k:])
        if any(n == m or n.startswith(m + ".") for n in nodes): return m
    return None

def gold_for_patch(patch, nodes, norm):
    gold_nodes, gold_files = set(), set()
    for path, hunks in parse_patch(patch).items():
        if hunks == "NEWFILE" or not path.endswith(".py"): continue
        if re.search(r"(^|/)(tests?|docs?|docs_src|scripts)/|(^|/)test_[^/]*\.py$|conftest", path): continue
        mod = module_prefixes(path, nodes)
        if not mod: continue
        gold_files.add(mod)
        cand = [n for n in nodes if n == mod or n.startswith(mod + ".")]
        for h in hunks:
            for probes in (h["removed"], h["context"]):
                P = []
                for l in probes:
                    w = words(l)
                    if len(w) >= 2 or (w and len(w[0]) >= 6): P.append(" " + " ".join(w) + " ")
                best, bn = [], 0
                for n in cand:
                    c = sum(1 for p in P if p in norm[n])
                    if c > bn: best, bn = [n], c
                    elif c == bn and c > 0: best.append(n)
                if bn:
                    gold_nodes.add(min(best, key=lambda n: len(nodes[n]))); break
            else:
                nm = re.search(r"(?:def|class)\s+(\w+)", h["header"])      # fall back to hunk header
                if nm:
                    hit = [n for n in cand if n.endswith("." + nm.group(1))]
                    if hit: gold_nodes.add(min(hit, key=len))
    return gold_nodes, gold_files

# ============================ retrievers ============================
def bm25(nodes, query, k1=1.4, b=0.75):
    ids = [n for n in nodes if nodes[n]]
    docs = {n: tokenize(n + " " + nodes[n]) for n in ids}
    df, tfs = Counter(), {}
    for n, d in docs.items():
        c = Counter(d); tfs[n] = c
        for t in c: df[t] += 1
    N = len(ids); avg = sum(len(d) for d in docs.values()) / max(1, N)
    q = Counter(tokenize(query)); sc = {}
    for n in ids:
        s, L = 0.0, max(1, len(docs[n]))
        for t, qf in q.items():
            f = tfs[n].get(t)
            if f:
                idf = math.log(1 + (N - df[t] + .5) / (df[t] + .5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg)) * (1 + math.log(qf))
        if s > 0: sc[n] = s
    return sc

def expand_graph(base, o, i, seeds=8, decay=0.35, hops=1):
    final = dict(base)
    fr = {s: base[s] for s in sorted(base, key=base.get, reverse=True)[:seeds]}
    for _ in range(hops):
        nxt = defaultdict(float)
        for n, s in fr.items():
            for nb in o.get(n, set()) | i.get(n, set()): nxt[nb] += s * decay
        for nb, s in nxt.items(): final[nb] = final.get(nb, 0) + s
        fr = nxt
    return final

def expand_emb(base, emb, seeds=5, per=5, decay=0.35):
    import numpy as np
    if emb is None: return dict(base)
    ids, M = emb; idx = {n: k for k, n in enumerate(ids)}; final = dict(base)
    for s in sorted(base, key=base.get, reverse=True)[:seeds]:
        if s not in idx: continue
        sims = M @ M[idx[s]]
        for j in np.argsort(-sims)[1:per + 1]:
            final[ids[j]] = final.get(ids[j], 0) + base[s] * decay * float(max(sims[j], 0))
    return final

def down_tests(sc):
    return {n: (s * 0.3 if re.search(r"(^|\.)(tests?|conftest)(\.|$)|\.test_", n) else s) for n, s in sc.items()}

METHODS = ["bm25", "bm25+graph", "bm25+graph(2hop)"] + (["bm25+emb", "bm25+graph+emb"] if WITH_EMB else [])
def run_method(m, base, g, emb):
    if m == "bm25": return base
    if m == "bm25+graph": return expand_graph(base, g[1], g[2])
    if m == "bm25+graph(2hop)": return expand_graph(base, g[1], g[2], hops=2)
    if m == "bm25+emb": return expand_emb(base, emb)
    return expand_graph(expand_emb(base, emb), g[1], g[2])

# ============================ evaluate ============================
KS = (1, 3, 5, 10)
node_hits = {m: Counter() for m in METHODS}; file_hits = {m: Counter() for m in METHODS}
by_repo = defaultdict(lambda: {m: Counter() for m in METHODS}); repo_n = Counter()
n_graph = n_node_lab = n_file_lab = 0
lab_rows = []
for t in TASKS:
    iid = t["instance_id"]; g = load_graph(iid)
    if g is None: continue
    n_graph += 1
    nodes = g[0]; norm = {n: " " + " ".join(words(txt)) + " " for n, txt in nodes.items()}
    gn, gf = gold_for_patch(t["patch"], nodes, norm)
    lab_rows.append([iid, t["repo"], ";".join(sorted(gf)), ";".join(sorted(gn))])
    if gf: n_file_lab += 1
    if gn: n_node_lab += 1
    if not gf: continue
    emb = load_emb(iid, nodes) if WITH_EMB else None
    base = down_tests(bm25(nodes, t["problem_statement"] + " " + (t.get("hints_text") or "")))
    repo_n[t["repo"]] += 1
    for m in METHODS:
        sc = down_tests(run_method(m, base, g, emb)) if m != "bm25" else base
        ranked = sorted(sc, key=sc.get, reverse=True)
        for k in KS:
            top = ranked[:k]
            if gn and gn & set(top): node_hits[m][k] += 1; by_repo[t["repo"]][m][("node", k)] += 1
            if any(r == f or r.startswith(f + ".") for r in top for f in gf):
                file_hits[m][k] += 1; by_repo[t["repo"]][m][("file", k)] += 1

with open("loc_labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["instance_id", "repo", "gold_files", "gold_nodes"]); w.writerows(lab_rows)

print(f"\ntasks with graph: {n_graph}; with gold FILE label: {n_file_lab}; with gold NODE label: {n_node_lab}")
n_f = max(1, n_file_lab); n_n = max(1, n_node_lab)
rows = [["method"] + [f"file@{k}" for k in KS] + [f"func@{k}" for k in KS]]
for m in METHODS:
    rows.append([m] + [f"{file_hits[m][k]/n_f:.3f}" for k in KS] + [f"{node_hits[m][k]/n_n:.3f}" for k in KS])
print(f"(file@k over {n_file_lab} tasks, func@k over {n_node_lab} tasks)")
for r in rows: print("  ".join(f"{c:>16}" for c in r))
csv.writer(open("loc_results.csv", "w", newline="")).writerows(rows)
rr = [["repo", "n", "method", "file@5", "func@5"]]
for repo, n in repo_n.items():
    for m in METHODS:
        rr.append([repo, n, m, f"{by_repo[repo][m][('file',5)]/n:.3f}", f"{by_repo[repo][m][('node',5)]/n:.3f}"])
csv.writer(open("loc_results_by_repo.csv", "w", newline="")).writerows(rr)
print("\nper repo @5:")
for r in rr[1:]: print("  ".join(f"{str(c):>16}" for c in r))
