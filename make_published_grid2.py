import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import audiometry as A

PATH = 'Review_Audiology_of_Published_Cases.xlsx'
FREQ_HZ = A.FREQ_HZ
BASE = "#1f4e79"                 # same blue as the cohort individual plots
ALPHA_FLOOR, ALPHA_TOP = 0.25, 1.0

def parse_cell(v):
    if pd.isna(v):
        return np.nan, None
    if isinstance(v, (int, float)):
        return float(v), None
    m = re.search(r'not recordable at\s*(\d+)', str(v).lower())
    if m:
        return np.nan, float(m.group(1))
    return np.nan, None

def load_published(path):
    xl = pd.ExcelFile(path)
    rows = []
    for s in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name=s, header=None)
        ear_row = raw.iloc[0].ffill()
        hdr = raw.iloc[1]
        freq_cols = {}
        for j, lab in hdr.items():
            if isinstance(lab, str) and lab in FREQ_HZ:
                ear = ear_row[j]
                ear = ear.replace(" Ear", "") if isinstance(ear, str) else None
                if ear in ("Right", "Left"):
                    freq_cols[j] = (ear, FREQ_HZ[lab])
        cage = [j for j, v in hdr.items() if str(v).strip().lower() == "age at test"][0]
        for _, r in raw.iloc[2:].iterrows():
            if pd.isna(r[cage]):
                continue
            for j, (ear, hz) in freq_cols.items():
                thr, nrl = parse_cell(r[j])
                rows.append({"participant": s, "age": float(r[cage]), "ear": ear,
                             "freq_hz": hz, "threshold": thr, "nr_level": nrl})
    return pd.DataFrame(rows)

df = load_published(PATH)

def figure_paired_grid(df, per_row=3):
    naud = df.groupby("participant")["age"].nunique()
    order = list(naud[naud > 1].sort_values(ascending=False).index) + \
            sorted(naud[naud == 1].index)
    n = len(order)
    nrows = int(np.ceil(n / per_row))
    amin, amax = df["age"].min(), df["age"].max()
    span = (amax - amin) or 1.0
    def alpha_for(a):
        return ALPHA_FLOOR + (ALPHA_TOP - ALPHA_FLOOR) * (a - amin) / span

    width = []
    for g in range(per_row):
        if g > 0:
            width.append(0.30)
        width += [1, 1]
    ncols = len(width)
    spacer_cols = {3 * g - 1 for g in range(1, per_row)}
    fig, axes = plt.subplots(nrows, ncols, figsize=(sum(width) * 2.7, 3.05 * nrows),
                             sharex=True, sharey=True,
                             gridspec_kw={"width_ratios": width})
    axes = np.atleast_2d(axes)

    col_bottom, used = {}, set()
    for i in range(n):
        row, g = i // per_row, i % per_row
        b = g * 3
        for c in (b, b + 1):
            col_bottom[c] = max(col_bottom.get(c, -1), row)
            used.add((row, c))

    def draw_ear(ax, de):
        for age, g in sorted(de.groupby("age")):
            g = g.sort_values("freq_hz")
            a = alpha_for(age)
            ax.plot(g["freq_hz"], g["threshold"], "-", color=BASE, alpha=a,
                    lw=1.1, zorder=2)
            pres = g[g["threshold"].notna()]
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"], marker="o", s=20,
                           color=BASE, alpha=a, edgecolors="none", zorder=3)
            nr = g[g["nr_level"].notna()]
            for _, rr in nr.iterrows():
                x, y = rr["freq_hz"], rr["nr_level"]
                ax.scatter([x], [y], marker="o", s=26, facecolors="none",
                           edgecolors=BASE, alpha=a, linewidths=1.3, zorder=5)
                ax.annotate("", xy=(x, y + 11), xytext=(x, y + 1.5),
                            arrowprops=dict(arrowstyle="-|>", color=BASE, alpha=a, lw=1.3),
                            zorder=5)

    for i, pid in enumerate(order):
        row, g = i // per_row, i % per_row
        b = g * 3
        ages = sorted(df[df.participant == pid]["age"].unique())
        agestr = ", ".join(f"{a:g}" for a in ages)
        for c, ear, tag in [(b, "Right", "R"), (b + 1, "Left", "L")]:
            ax = axes[row, c]
            draw_ear(ax, df[(df.participant == pid) & (df.ear == ear)])
            A._audiogram_xaxis(ax); A._db_yaxis(ax)
            ax.set_title(f"{tag}   age {agestr}", fontsize=8.5)
            if c != 0:
                ax.set_ylabel(""); ax.tick_params(labelleft=False)
            if row != col_bottom[c]:
                ax.set_xlabel(""); ax.tick_params(labelbottom=False)

    for r in range(nrows):
        for c in range(ncols):
            if c in spacer_cols or (r, c) not in used:
                axes[r, c].axis("off")

    handles = [
        Line2D([0], [0], marker="o", color=BASE, lw=1.1, label="Measured threshold"),
        Line2D([0], [0], marker="o", color=BASE, lw=0, markerfacecolor="none",
               markeredgecolor=BASE, label="No response (arrow = \u2265 level)"),
        Line2D([0], [0], color=BASE, lw=4, alpha=ALPHA_FLOOR, label=f"Youngest ({amin:g} yr)"),
        Line2D([0], [0], color=BASE, lw=4, alpha=ALPHA_TOP, label=f"Oldest ({amax:g} yr)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Published Norrie disease audiograms by case \u2014 Right | Left paired "
                 "(opacity \u221d age at test)", y=0.998, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    return fig

fig = figure_paired_grid(df)
fig.savefig('figM_published_paired_grid_blue.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("saved; age range", df.age.min(), "-", df.age.max())
