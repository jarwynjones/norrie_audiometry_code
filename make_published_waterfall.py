import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import audiometry as A

PATH = 'Review_Audiology_of_Published_Cases.xlsx'

# ---- custom parse: capture "not recordable at N" as (level=N, no_response=True) ----
FREQ_HZ = A.FREQ_HZ
def parse_cell(v):
    """Return (threshold, nr_level). Numeric -> (float, None). 'not recordable at N' ->
    (nan, N) meaning no response at level N. Blank/other -> (nan, None)."""
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

CASE_MARKER = {'Exon 2 del ND44': 's', 'Parving 1977 no mutation': 'D'}
SERIAL = set(CASE_MARKER)

def figure_published(df):
    ages = df['age'].dropna()
    amin, amax = ages.min(), ages.max()
    norm = plt.Normalize(amin, amax)
    cmap = LinearSegmentedColormap.from_list(
        "age_ref", ["#6cbf43", "#1aa07a", "#1f9bd0", "#5a6fc0", "#9b5fb8", "#e85aa0"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, ear in zip(axes, ["Right", "Left"]):
        de = df[df.ear == ear]
        for (pid, age), g in sorted(de.groupby(["participant", "age"]), key=lambda kv: kv[0][1]):
            g = g.sort_values("freq_hz")
            color = cmap(norm(age))
            marker = CASE_MARKER.get(pid, "o")
            serial = pid in SERIAL
            # connect measured thresholds (line breaks at NaN)
            ax.plot(g["freq_hz"], g["threshold"], "-", color=color, alpha=0.5, lw=0.9, zorder=2)
            pres = g[g["threshold"].notna()]
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"], marker=marker,
                           s=34 if serial else 26, color=color,
                           alpha=0.85 if serial else 0.6, edgecolors="none",
                           zorder=4 if serial else 3)
            # no-response points: case marker at stated level + downward arrow
            nr = g[g["nr_level"].notna()]
            for _, row in nr.iterrows():
                x, y = row["freq_hz"], row["nr_level"]
                ax.scatter([x], [y], marker=marker, s=40 if serial else 32,
                           facecolors="none", edgecolors=color, linewidths=1.4, zorder=5)
                ax.annotate("", xy=(x, y + 11), xytext=(x, y + 1.5),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4), zorder=5)
        A._audiogram_xaxis(ax); A._db_yaxis(ax)
        ax.set_title(f"{ear} ear")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Age at test (years)")
    shape_handles = [
        Line2D([0], [0], marker="o", color="grey", lw=0.9, markerfacecolor="grey",
               label="Single-timepoint case"),
        Line2D([0], [0], marker="s", color="grey", lw=0.9, markerfacecolor="grey",
               label="Serial: Exon 2 del (ND44)"),
        Line2D([0], [0], marker="D", color="grey", lw=0.9, markerfacecolor="grey",
               label="Serial: Parving 1977"),
        Line2D([0], [0], marker=r"$\downarrow$", color="grey", lw=0, markersize=11,
               label="No response (arrow = \u2265 level)"),
    ]
    axes[1].legend(handles=shape_handles, loc="lower right", fontsize=8.5, framealpha=0.9)
    n_case = df.participant.nunique()
    n_audio = df[["participant", "age"]].drop_duplicates().shape[0]
    fig.suptitle(f"Published Norrie disease audiograms \u2014 {n_case} cases, {n_audio} audiograms "
                 f"(colour = age; digitised from literature)", y=0.99, fontsize=11)
    return fig

figure_published(df).savefig('figH_published_waterfall.png', dpi=150, bbox_inches='tight')
plt.close('all')
nr = df[df.nr_level.notna()]
print("no-response points plotted:", len(nr))
print(nr[["participant","age","ear","freq_hz","nr_level"]].to_string(index=False))
df.to_csv('published_long.csv', index=False)
