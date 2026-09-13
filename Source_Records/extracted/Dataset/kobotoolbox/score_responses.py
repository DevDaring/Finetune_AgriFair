"""score_responses.py — analyse the two annotators' KoboToolbox responses.

Compares each annotator's AgriFacts answers to the census-computed gold answers,
checks AgriAdvice pair confirmations (expected 'Yes'), and reports inter-annotator
agreement. Labels-format export (semicolon-delimited).
"""
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
CSV = HERE / "AgriFair_-_human_verification_-_all_versions_-_labels_-_2026-06-22-06-49-56.csv"
KEY = HERE / "AgriFair_answer_key.xlsx"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


df = pd.read_csv(CSV, sep=";", dtype=str, keep_default_na=False)
key = pd.read_excel(KEY, dtype=str)
names = list(df["Your name or initials"])
A, B = names[0], names[1]

fkey = {norm(r["prompt"]): (norm(r["expected_answer"]), r["axis"], r["condition"])
        for _, r in key[key.section == "agrifacts"].iterrows()}
adv_axis = [(norm(r["prompt"]), r["axis"]) for _, r in key[key.section == "agriadvice"].iterrows()]

qcols = [c for c in df.columns if re.match(r"^Q\d+\.", c)]
pcols = [c for c in df.columns if re.match(r"^P\d+\.", c)]


def yn(v):
    v = str(v).lower()
    if v.startswith("yes"):
        return "yes"
    if v.startswith("no"):
        return "no"
    if v.startswith("unsure"):
        return "unsure"
    return "blank"


# ---------------- AgriFacts ----------------
rows = []
for c in qcols:
    qt = norm(re.sub(r"^Q\d+\.\s*", "", c))
    gold, axis, cond = fkey[qt]
    a, b = norm(df.iloc[0][c]), norm(df.iloc[1][c])
    rows.append({"axis": axis, "cond": cond, "gold": gold, "a": a, "b": b,
                 "a_ok": a == gold, "b_ok": b == gold, "ab_same": a == b})
fdf = pd.DataFrame(rows)

print("=" * 64)
print("ANNOTATORS:", f"A = {A}", "|", f"B = {B}")
print(f"Items: AgriFacts = {len(qcols)} MCQs, AgriAdvice = {len(pcols)} pairs")
print("=" * 64)

print("\n--- PART A: AgriFacts (accuracy vs census-computed gold) ---")
print(f"  {A:22s}: {fdf.a_ok.sum():3d}/{len(fdf)} = {fdf.a_ok.mean():.1%}")
print(f"  {B:22s}: {fdf.b_ok.sum():3d}/{len(fdf)} = {fdf.b_ok.mean():.1%}")
either = (fdf.a_ok | fdf.b_ok).mean()
both = (fdf.a_ok & fdf.b_ok).mean()
print(f"  at least one correct : {either:.1%}")
print(f"  both correct         : {both:.1%}")

print("\n  accuracy by axis (A / B):")
for ax, g in fdf.groupby("axis"):
    print(f"    {ax:14s}: {g.a_ok.mean():.0%} / {g.b_ok.mean():.0%}   (n={len(g)})")
print("  accuracy by condition (A / B):")
for cd, g in fdf.groupby("cond"):
    print(f"    {cd:14s}: {g.a_ok.mean():.0%} / {g.b_ok.mean():.0%}   (n={len(g)})")

# inter-annotator agreement + Cohen's kappa on AgriFacts choices
po = fdf.ab_same.mean()
# expected agreement from marginal choice distributions
pa = pd.Series(fdf.a).value_counts(normalize=True)
pb = pd.Series(fdf.b).value_counts(normalize=True)
pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in set(pa.index) | set(pb.index))
kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
print(f"\n  inter-annotator agreement (same option): {po:.1%}   Cohen's kappa = {kappa:.2f}")

# items BOTH got wrong -> worth inspecting in the dataset
both_wrong = fdf[(~fdf.a_ok) & (~fdf.b_ok)]
print(f"\n  items BOTH annotators answered differently from gold: {len(both_wrong)}")
for _, r in both_wrong.head(12).iterrows():
    print(f"    [{r.axis}/{r.cond}] gold={r.gold!r}  A={r.a!r}  B={r.b!r}")

# ---------------- AgriAdvice ----------------
padf = []
for i, c in enumerate(pcols):
    a, b = yn(df.iloc[0][c]), yn(df.iloc[1][c])
    # axis by base_query substring match
    axis = next((ax for q, ax in adv_axis if q and q in norm(c)), "?")
    padf.append({"axis": axis, "a": a, "b": b})
padf = pd.DataFrame(padf)

print("\n--- PART B: AgriAdvice (expected 'Yes' = pair preserves content) ---")
for who, col in [(A, "a"), (B, "b")]:
    vc = padf[col].value_counts().to_dict()
    yes = vc.get("yes", 0)
    print(f"  {who:22s}: Yes={yes}/{len(padf)} = {yes/len(padf):.0%}   {vc}")
po2 = (padf.a == padf.b).mean()
print(f"  inter-annotator agreement (same yes/no): {po2:.1%}")
print("  'Yes' rate by axis (A / B):")
for ax, g in padf.groupby("axis"):
    print(f"    {ax:18s}: {(g.a=='yes').mean():.0%} / {(g.b=='yes').mean():.0%}   (n={len(g)})")

flagged = padf[(padf.a != "yes") | (padf.b != "yes")]
print(f"\n  pairs flagged (either annotator not 'Yes'): {len(flagged)}")
for i, r in flagged.iterrows():
    print(f"    {pcols[i][:46].strip()}...  A={r.a} B={r.b}  axis={r.axis}")
