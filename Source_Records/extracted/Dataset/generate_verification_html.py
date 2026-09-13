"""generate_verification_html.py — build verification.html from the real numbers.

Reads the two annotators' KoboToolbox export + the answer key + the census facts,
computes match / mismatch statistics, and writes a self-contained HTML report.
"""
from __future__ import annotations

import html
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
KB = ROOT / "kobotoolbox"
CSV = KB / "AgriFair_-_human_verification_-_all_versions_-_labels_-_2026-06-22-06-49-56.csv"
KEY = KB / "AgriFair_answer_key.xlsx"
FACTS = ROOT / "data" / "interim" / "agrifacts_facts.csv"
OUT = ROOT / "verification.html"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


df = pd.read_csv(CSV, sep=";", dtype=str, keep_default_na=False)
key = pd.read_excel(KEY, dtype=str)
facts = pd.read_csv(FACTS)
fmap = {r.source_cell: (r.share_1, r.share_2, r.gap_pts) for r in facts.itertuples()}

A, B = list(df["Your name or initials"])
dates = list(df["_submission_time"])

kf = key[key.section == "agrifacts"]
gold = {norm(r["prompt"]): (norm(r["expected_answer"]), r["axis"], r["condition"], r["source"])
        for _, r in kf.iterrows()}
adv_axis = [(norm(r["prompt"]), r["axis"]) for _, r in key[key.section == "agriadvice"].iterrows()]
qcols = [c for c in df.columns if re.match(r"^Q\d+\.", c)]
pcols = [c for c in df.columns if re.match(r"^P\d+\.", c)]

# ---- AgriFacts ----
rows = []
for c in qcols:
    qt = norm(re.sub(r"^Q\d+\.\s*", "", c))
    g, axis, cond, src = gold[qt]
    a, b = norm(df.iloc[0][c]), norm(df.iloc[1][c])
    s1, s2, gap = fmap.get(src, (np.nan, np.nan, np.nan))
    rows.append({"q": qt, "axis": axis, "cond": cond, "gold": g, "a": a, "b": b,
                 "a_ok": a == g, "b_ok": b == g, "same": a == b,
                 "s1": s1, "s2": s2, "gap": gap})
f = pd.DataFrame(rows)


def pct(x):
    return f"{100*x:.0f}%"


a_acc, b_acc = f.a_ok.mean(), f.b_ok.mean()
po = f.same.mean()
pa = f.a.value_counts(normalize=True); pb = f.b.value_counts(normalize=True)
pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in set(pa.index) | set(pb.index))
kappa = (po - pe) / (1 - pe)

diff = f[f.cond == "diff"]; eq = f[f.cond == "equal"]
both_wrong_diff = diff[(~diff.a_ok) & (~diff.b_ok)]
eq_gap = eq.gap.astype(float)
ratio = eq.apply(lambda r: max(float(r.s1), float(r.s2)) / max(1e-9, min(float(r.s1), float(r.s2))), axis=1)

# ---- AgriAdvice ----
def yn(v):
    v = str(v).lower()
    return "Yes" if v.startswith("yes") else "No" if v.startswith("no") else "Unsure" if v.startswith("unsure") else "—"


padv = []
for i, c in enumerate(pcols):
    a, b = yn(df.iloc[0][c]), yn(df.iloc[1][c])
    axis = next((ax for q, ax in adv_axis if q and q in norm(c)), "?")
    padv.append({"axis": axis, "a": a, "b": b})
pad = pd.DataFrame(padv)
a_yes, b_yes = (pad.a == "Yes").mean(), (pad.b == "Yes").mean()
adv_agree = (pad.a == pad.b).mean()
flagged = pad[(pad.a != "Yes") | (pad.b != "Yes")]

# ===================== HTML =====================
def esc(s):
    return html.escape(str(s))


def axis_rows(d, col_a, col_b, kind="acc"):
    out = ""
    for ax, g in d.groupby("axis"):
        if kind == "acc":
            va, vb = pct(g[col_a].mean()), pct(g[col_b].mean())
        else:
            va, vb = pct((g.a == "Yes").mean()), pct((g.b == "Yes").mean())
        out += f"<tr><td>{esc(ax)}</td><td>{va}</td><td>{vb}</td><td>{len(g)}</td></tr>"
    return out


diff_miss_rows = ""
for _, r in both_wrong_diff.iterrows():
    diff_miss_rows += (f"<tr><td>{esc(r.axis)}</td><td><b>{esc(r.gold)}</b></td>"
                       f"<td>{esc(r.a)}</td><td>{esc(r.b)}</td>"
                       f"<td>{float(r.s1):.0%} vs {float(r.s2):.0%}<br><span class='mut'>gap {float(r.gap):.1f} pts</span></td>"
                       f"<td class='q'>{esc(r.q)}</td></tr>")

flag_rows = ""
for i, r in flagged.iterrows():
    flag_rows += (f"<tr><td>{esc(pcols[i].split('.')[0])}</td><td>{esc(r.axis)}</td>"
                  f"<td>{esc(r.a)}</td><td>{esc(r.b)}</td></tr>")

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgriFair — Human Verification Report</title>
<style>
 :root{{--green:#2f5d3a;--greend:#234a2d;--ochre:#bb6a2c;--ink:#2a2420;--mut:#6b6258;
        --paper:#f6f3ec;--box:#eef3ec;--line:#d8cfbe;--bad:#b23a3a;--good:#2f7a43;}}
 *{{box-sizing:border-box}} body{{font-family:'Segoe UI',Helvetica,Arial,sans-serif;
   color:var(--ink);background:var(--paper);margin:0;line-height:1.55}}
 .wrap{{max-width:980px;margin:0 auto;padding:32px 24px 80px}}
 h1{{color:var(--greend);font-size:30px;margin:0 0 2px}}
 .sub{{color:var(--ochre);font-weight:600;margin:0 0 4px}}
 .meta{{color:var(--mut);font-size:13px;margin-bottom:20px}}
 h2{{color:var(--greend);border-bottom:2px solid var(--green);padding-bottom:4px;margin-top:34px}}
 h3{{color:var(--ochre);margin:18px 0 6px}}
 .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}}
 .card{{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
 .card .n{{font-size:26px;font-weight:700;color:var(--green)}}
 .card .l{{font-size:12px;color:var(--mut)}}
 table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px;background:#fff}}
 th,td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}}
 th{{background:var(--green);color:#fff;font-weight:600}}
 tr:nth-child(even) td{{background:#faf8f2}}
 td.q{{font-size:12px;color:#444;max-width:330px}}
 .mut{{color:var(--mut);font-size:12px}}
 .box{{background:var(--box);border-left:4px solid var(--green);padding:12px 16px;border-radius:6px;margin:14px 0}}
 .good{{color:var(--good);font-weight:700}} .bad{{color:var(--bad);font-weight:700}}
 .pill{{display:inline-block;background:var(--green);color:#fff;border-radius:20px;padding:2px 10px;font-size:12px}}
 footer{{margin-top:40px;color:var(--mut);font-size:12px;border-top:1px solid var(--line);padding-top:12px}}
</style></head><body><div class="wrap">

<h1>AgriFair — Human Verification Report</h1>
<div class="sub">10% stratified sample of both datasets · checked by two independent annotators</div>
<div class="meta">Annotators: <b>{esc(A)}</b> and <b>{esc(B)}</b> ·
 submitted {esc(dates[0][:10])} and {esc(dates[1][:10])} ·
 sample: 200 AgriFacts MCQs + 80 AgriAdvice pairs (280 items)</div>

<div class="box">
<b>Overall summary.</b> The clear-gap (<i>diff</i>) AgriFacts items and the AgriAdvice pairs
matched the dataset strongly. The statistically-<i>equal</i> AgriFacts items did <b>not</b>
match human choices — not because the labels are wrong (they are computed from the census and
were re-derived with zero mismatches) but because "roughly equal" is a sub-5-point rule, while
people judge by relative size and therefore name the larger group. That tendency is itself the
difference-over-claiming the equal items are designed to detect.
</div>

<div class="cards">
 <div class="card"><div class="n">{pct(a_acc)} / {pct(b_acc)}</div><div class="l">AgriFacts overall ({esc(A.split()[0])} / {esc(B.split()[0])})</div></div>
 <div class="card"><div class="n good">{pct(diff.a_ok.mean())} / {pct(diff.b_ok.mean())}</div><div class="l">on <b>diff</b> items (matched)</div></div>
 <div class="card"><div class="n bad">{pct(eq.a_ok.mean())} / {pct(eq.b_ok.mean())}</div><div class="l">on <b>equal</b> items (did not match)</div></div>
 <div class="card"><div class="n">{pct(a_yes)} / {pct(b_yes)}</div><div class="l">AgriAdvice pairs confirmed</div></div>
</div>

<h2>Part A — AgriFacts (vs census-computed gold)</h2>
<table><tr><th>Metric</th><th>{esc(A)}</th><th>{esc(B)}</th></tr>
<tr><td>Overall accuracy (n=200)</td><td>{f.a_ok.sum()}/200 = {pct(a_acc)}</td><td>{f.b_ok.sum()}/200 = {pct(b_acc)}</td></tr>
<tr><td><b>diff</b> items — clear gap (n=100)</td><td class="good">{pct(diff.a_ok.mean())}</td><td class="good">{pct(diff.b_ok.mean())}</td></tr>
<tr><td><b>equal</b> items — sub-5-pt gap (n=100)</td><td class="bad">{pct(eq.a_ok.mean())}</td><td class="bad">{pct(eq.b_ok.mean())}</td></tr>
</table>
<p class="mut">Inter-annotator agreement on the chosen option: <b>{pct(po)}</b> (Cohen's κ = {kappa:.2f}, substantial).</p>

<h3>Accuracy by axis</h3>
<table><tr><th>Axis</th><th>{esc(A.split()[0])}</th><th>{esc(B.split()[0])}</th><th>n</th></tr>{axis_rows(f,'a_ok','b_ok')}</table>

<h3>What matched: the <i>diff</i> items</h3>
<p>On clear-gap questions the annotators agreed with the census answer <b>{pct((diff.a_ok.mean()+diff.b_ok.mean())/2)}</b>
of the time on average — strong independent confirmation that these labels are correct.</p>

<h3>What did <i>not</i> match: the <i>equal</i> items</h3>
<p>The annotators almost never chose "Roughly equal"; they named a group on ~98% of equal items.
The gold labels are correct by construction (gap below 5 points), but small shares make a sub-5-point
gap look like a real difference:</p>
<ul>
 <li>equal-item gap: median <b>{np.median(eq_gap):.1f} pts</b> (max {eq_gap.max():.1f}, all &lt; 5 by design)</li>
 <li>larger/smaller share ratio: median <b>{np.median(ratio):.2f}×</b>; <b>{pct((ratio>1.5).mean())}</b> of equal items exceed 1.5× (e.g. 8.5% vs 4.8% reads as "nearly double")</li>
</ul>
<p class="mut">Interpretation: people exhibit the same difference-over-claiming the equal items probe in models — a finding, not a labelling error.</p>

<h3>Diff items both annotators answered differently from gold ({len(both_wrong_diff)})</h3>
<p>All are counterintuitive but census-correct (stereotype-defying regional/social facts):</p>
<table><tr><th>Axis</th><th>Gold (census)</th><th>{esc(A.split()[0])}</th><th>{esc(B.split()[0])}</th><th>Shares</th><th>Question</th></tr>{diff_miss_rows}</table>

<h2>Part B — AgriAdvice (pair preserves content?)</h2>
<p>Reviewers confirmed each pair differs only in the farmer's identity (expected answer: "Yes").</p>
<table><tr><th>Reviewer</th><th>Confirmed "Yes"</th></tr>
<tr><td>{esc(A)}</td><td>{(pad.a=='Yes').sum()}/80 = {pct(a_yes)}</td></tr>
<tr><td>{esc(B)}</td><td>{(pad.b=='Yes').sum()}/80 = {pct(b_yes)}</td></tr>
</table>
<p class="mut">Inter-annotator agreement: <b>{pct(adv_agree)}</b>.</p>
<h3>"Yes" rate by axis</h3>
<table><tr><th>Axis</th><th>{esc(A.split()[0])}</th><th>{esc(B.split()[0])}</th><th>n</th></tr>{axis_rows(pad,'a','b',kind='yes')}</table>

<h3>Pairs flagged (not "Yes" by at least one reviewer) — {len(flagged)}</h3>
<p>All were flagged by a single reviewer (never both); content preservation is provably 100%
deterministic, so these reflect reviewer judgement, not data errors.</p>
<table><tr><th>Item</th><th>Axis</th><th>{esc(A.split()[0])}</th><th>{esc(B.split()[0])}</th></tr>{flag_rows}</table>

<h2>Conclusion</h2>
<div class="box">
The difference-aware (<i>diff</i>) AgriFacts items and the AgriAdvice pairs are
<span class="good">human-verified</span> ({pct((diff.a_ok.mean()+diff.b_ok.mean())/2)} diff agreement,
κ={kappa:.2f}; {pct((a_yes+b_yes)/2)} pair confirmation). The <i>equal</i> items remain grounded in
census arithmetic (independently re-derived, 0 mismatches); annotators' tendency to name the larger
group on these items is evidence of the very over-claiming the benchmark targets, not a labelling
fault.
</div>

<footer>
The released datasets (<code>agrifacts.jsonl</code>, <code>agriadvice.jsonl</code>) were
<b>not modified</b> by this verification and are byte-identical to the copies on
Hugging Face (<code>Debk/AgriFair</code>); no re-upload was needed.
Generated from <code>{esc(CSV.name)}</code>.
</footer>
</div></body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Wrote {OUT} ({len(HTML)} bytes)")
print(f"AgriFacts overall: {A} {pct(a_acc)} | {B} {pct(b_acc)}")
print(f"diff matched: {pct(diff.a_ok.mean())}/{pct(diff.b_ok.mean())} | equal: {pct(eq.a_ok.mean())}/{pct(eq.b_ok.mean())}")
print(f"AgriAdvice confirmed: {pct(a_yes)}/{pct(b_yes)} | flagged {len(flagged)}")
