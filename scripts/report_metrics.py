"""Re-derive every number cited in §2 + §3 of the report. Run before final draft."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

base = Path("experiments/studies")
fams: Counter[str] = Counter()
rows: list[dict] = []
kaggle: list[dict] = []
for d in sorted(base.iterdir()):
    f = d / "study.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())
    rows.append({
        "id": s.get("id", d.name),
        "status": s.get("status"),
        "n_exp": len(s.get("experiments") or []),
        "best_val": s.get("best_score"),
    })
    for e in s.get("experiments") or []:
        prop = e.get("proposal") or {}
        fams[prop.get("family") or prop.get("architecture_name") or "?"] += 1
    for ks in s.get("kaggle_scores") or []:
        kaggle.append({"study": s.get("id", d.name), **ks})

best_val = max((r["best_val"] for r in rows if r["best_val"] is not None), default=None)
public = [k for k in kaggle if k.get("leaderboard") == "public"]
private = [k for k in kaggle if k.get("leaderboard") == "private"]
best_public = max(public, key=lambda k: k["score"], default=None)
best_private = max(private, key=lambda k: k["score"], default=None)

print(f"studies={len(rows)} completed={sum(1 for r in rows if r['status']=='COMPLETED')}")
print(f"experiments={sum(r['n_exp'] for r in rows)} best_val_macro_auc={best_val:.4f}")
if best_public:
    print(f"best_kaggle_public={best_public['score']} (study={best_public['study']} exp={best_public['experiment_id']})")
if best_private:
    print(f"best_kaggle_private={best_private['score']} (study={best_private['study']} exp={best_private['experiment_id']})")
print("\nFamily distribution:")
for k, v in fams.most_common():
    print(f"  {v:3d}  {k}")
print("\nAll Kaggle scores:")
for k in sorted(kaggle, key=lambda x: -x["score"]):
    print(f"  {k['score']:.3f} {k['leaderboard']:8s} {k['study']}/{k['experiment_id']}")
