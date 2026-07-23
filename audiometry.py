"""
Serial pure-tone audiometry: ingestion + figures.
Per-sheet (per-participant) functions; scales to a multi-sheet workbook unchanged.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# ---- configuration -------------------------------------------------------
FREQ_HZ = {"125Hz": 125, "250Hz": 250, "500Hz": 500, "1kHz": 1000,
           "2kHz": 2000, "4kHz": 4000, "8kHz": 8000}
FREQ_ORDER = sorted(FREQ_HZ.values())
FREQ_LABEL = {125: "125", 250: "250", 500: "500", 1000: "1k",
              2000: "2k", 4000: "4k", 8000: "8k"}

# test "Type" -> conventional PTA (circle) vs behavioural/other (triangle)
def classify_test(t):
    return "PTA" if isinstance(t, str) and t.strip().lower().startswith("pure tone audiometry") else "Other"

CONDUCTIVE_KEYWORDS = ["etd", "middle ear", "effusion", "negative", "flat", "type", "cold", "illness"]

def flag_comment(c):
    if not isinstance(c, str):
        return False, ""
    low = c.lower()
    hits = [k for k in CONDUCTIVE_KEYWORDS if k in low]
    return (len(hits) > 0), ", ".join(hits)

Y_MIN, Y_MAX = -10, 120          # standard clinical audiogram range
ALPHA_FLOOR, ALPHA_TOP = 0.25, 1.0
EXCEL_EPOCH = "1899-12-30"
NR_TEXT = "no response"          # case-insensitive match
NR_LEVEL = 120                   # dB HL placement for no-response symbol

def parse_threshold(v):
    """Return (numeric_threshold, is_no_response).
    Numeric -> (float, False). 'No response' -> (np.nan, True).
    Anything else non-numeric (e.g. a range like '30-35') -> (np.nan, False) i.e. treated as missing.
    """
    if pd.isna(v):
        return np.nan, False
    if isinstance(v, (int, float)):
        return float(v), False
    s = str(v).strip().lower()
    if s == NR_TEXT:
        return np.nan, True
    return np.nan, False          # unrecognised non-numeric -> missing

# ---- ingestion -----------------------------------------------------------
def load_sheet(path, sheet):
    """Parse one double-headed participant sheet into long format."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    ear_row = raw.iloc[0].ffill()          # forward-fill 'Right Ear'/'Left Ear'
    hdr_row = raw.iloc[1]
    # map each data column to (ear, freq_hz)
    freq_cols = {}
    for j, label in hdr_row.items():
        if isinstance(label, str) and label in FREQ_HZ:
            ear = ear_row[j]
            ear = ear.replace(" Ear", "") if isinstance(ear, str) else None
            if ear in ("Right", "Left"):
                freq_cols[j] = (ear, FREQ_HZ[label])
    # locate the meta columns by header name
    def col_of(name):
        m = [j for j, v in hdr_row.items() if isinstance(v, str) and v.strip().lower() == name]
        return m[0] if m else None
    c_date, c_age, c_type, c_comment = (col_of("date"), col_of("age at test"),
                                        col_of("type"), col_of("comment"))
    body = raw.iloc[2:].reset_index(drop=True)
    rows = []
    for _, r in body.iterrows():
        if pd.isna(r[c_date]) and pd.isna(r[c_age]):
            continue
        flagged, hits = flag_comment(r[c_comment])
        dv = r[c_date]
        if pd.isna(dv):
            date = pd.NaT
        elif isinstance(dv, (int, float)):
            date = pd.to_datetime(dv, origin=EXCEL_EPOCH, unit="D")
        else:
            date = pd.to_datetime(dv)
        for j, (ear, hz) in freq_cols.items():
            thr, nr = parse_threshold(r[j])
            rows.append({
                "participant": sheet,
                "date": date,
                "age": float(r[c_age]) if pd.notna(r[c_age]) else np.nan,
                "test_type": r[c_type],
                "test_class": classify_test(r[c_type]),
                "ear": ear,
                "freq_hz": hz,
                "threshold": thr,
                "no_response": nr,
                "flagged": flagged,
                "flag_hits": hits,
                "comment": r[c_comment] if pd.notna(r[c_comment]) else "",
            })
    return pd.DataFrame(rows)

EXPECTED_HDR = ('Date', 'Age at test', 'Type', '', '125Hz', '250Hz', '500Hz', '1kHz',
                '2kHz', '4kHz', '8kHz', 'Pure tone average (4 frequency method)', '',
                '125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz', '8kHz',
                'Pure tone average (4 frequency method)', 'Comment')

def validate_workbook(path):
    """Fail loudly if any sheet deviates from the expected two-row layout."""
    xl = pd.ExcelFile(path)
    problems = []
    for s in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name=s, header=None, nrows=2)
        hdr = tuple('' if pd.isna(x) else str(x).strip() for x in raw.iloc[1])
        if hdr != EXPECTED_HDR:
            problems.append(f"  {s}: header row mismatch -> {hdr}")
    if problems:
        raise ValueError("Structural validation FAILED:\n" + "\n".join(problems))
    return xl.sheet_names

def load_workbook(path):
    validate_workbook(path)
    xl = pd.ExcelFile(path)
    return pd.concat([load_sheet(path, s) for s in xl.sheet_names], ignore_index=True)

def flag_table(df):
    """One row per test, for manual verification of the conductive flag."""
    g = (df[["participant", "date", "age", "test_type", "flagged", "flag_hits", "comment"]]
         .drop_duplicates(subset=["participant", "date"])
         .sort_values(["participant", "age"]).reset_index(drop=True))
    return g

# ---- shared axis styling -------------------------------------------------
def _audiogram_xaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim(100, 10000)
    ax.xaxis.set_major_locator(FixedLocator(FREQ_ORDER))
    ax.xaxis.set_major_formatter(FixedFormatter([FREQ_LABEL[f] for f in FREQ_ORDER]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("Frequency (Hz)")

def _db_yaxis(ax):
    ax.set_ylim(NR_LEVEL + 8, Y_MIN)     # small headroom below 120 so NR boxed-crosses render fully
    ax.set_yticks(range(0, Y_MAX + 1, 10))
    ax.set_ylabel("Hearing threshold (dB HL)")
    ax.grid(True, which="major", lw=0.4, alpha=0.4)

# ---- Figure A: audiogram waterfall, opacity = age ------------------------
def figure_waterfall(df, participant, base_color="#1f4e79", flag_color="#d62728"):
    d = df[df.participant == participant]
    ages = d["age"].dropna()
    amin, amax = ages.min(), ages.max()
    def alpha_for(a):
        if amax == amin:
            return ALPHA_TOP
        return ALPHA_FLOOR + (ALPHA_TOP - ALPHA_FLOOR) * (a - amin) / (amax - amin)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, ear in zip(axes, ["Right", "Left"]):
        de = d[d.ear == ear]
        for (date, age), g in sorted(de.groupby(["date", "age"]), key=lambda kv: kv[0][1]):
            g = g.sort_values("freq_hz")
            x = g["freq_hz"].to_numpy()
            y = g["threshold"].to_numpy()           # np.nan -> line break
            a = alpha_for(age)
            ax.plot(x, y, "-", color=base_color, alpha=a, lw=1.1, zorder=2)
            present = g[g["threshold"].notna()]
            is_pta = g["test_class"].iloc[0] == "PTA"
            flagged = bool(g["flagged"].iloc[0])
            if len(present):
                ax.scatter(present["freq_hz"], present["threshold"],
                           marker=("o" if is_pta else "^"),
                           s=26, color=base_color, alpha=a,
                           edgecolors=(flag_color if flagged else "none"),
                           linewidths=(1.4 if flagged else 0), zorder=3)
            # no-response: boxed cross at NR_LEVEL (line already broken via NaN threshold)
            nr = g[g["no_response"]]
            if len(nr):
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="s", s=70,
                           facecolors="none", edgecolors=base_color, alpha=a,
                           linewidths=1.1, zorder=4)
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="x", s=34,
                           color=base_color, alpha=a, linewidths=1.1, zorder=4)
                if flagged:
                    ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="s", s=110,
                               facecolors="none", edgecolors=flag_color, alpha=a,
                               linewidths=1.3, zorder=5)
        _audiogram_xaxis(ax); _db_yaxis(ax)
        ax.set_title(f"{ear} ear")

    handles = [
        Line2D([0], [0], marker="o", color=base_color, lw=1.1, label="Pure-tone audiometry"),
        Line2D([0], [0], marker="^", color=base_color, lw=0, label="Play / behavioural"),
        Line2D([0], [0], marker="o", color=base_color, lw=0, markeredgecolor=flag_color,
               markeredgewidth=1.4, label="Conductive flag (verify)"),
        Line2D([0], [0], marker="s", color=base_color, lw=0, markerfacecolor="none",
               markersize=9, label=f"No response (plotted at {NR_LEVEL} dB)"),
        Line2D([0], [0], color=base_color, lw=4, alpha=ALPHA_FLOOR, label=f"Youngest ({amin:.1f} yr)"),
        Line2D([0], [0], color=base_color, lw=4, alpha=ALPHA_TOP, label=f"Oldest ({amax:.1f} yr)"),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Serial audiograms — participant {participant} (opacity ∝ age)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ---- Figure B: per-frequency trajectories vs age -------------------------
def figure_trajectories(df, participant, flag_color="#d62728"):
    d = df[df.participant == participant]
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(FREQ_ORDER)))
    fcolor = {f: cmap[i] for i, f in enumerate(FREQ_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, ear in zip(axes, ["Right", "Left"]):
        de = d[d.ear == ear]
        for f in FREQ_ORDER:
            gf = de[de.freq_hz == f].sort_values("age")
            x = gf["age"].to_numpy()
            y = gf["threshold"].to_numpy()          # np.nan -> line break
            ax.plot(x, y, "-", color=fcolor[f], lw=1.1, alpha=0.9, zorder=2)
            pres = gf[gf["threshold"].notna()]
            for cls, mk in [("PTA", "o"), ("Other", "^")]:
                sub = pres[pres.test_class == cls]
                if not len(sub):
                    continue
                edge = [flag_color if fl else fcolor[f] for fl in sub["flagged"]]
                lw = [1.4 if fl else 0.5 for fl in sub["flagged"]]
                ax.scatter(sub["age"], sub["threshold"], marker=mk, s=30,
                           facecolors=[fcolor[f]] * len(sub),
                           edgecolors=edge, linewidths=lw, zorder=3)
            # no-response: boxed cross at NR_LEVEL, coloured by frequency, line broken via NaN
            nr = gf[gf["no_response"]]
            if len(nr):
                ax.scatter(nr["age"], [NR_LEVEL] * len(nr), marker="s", s=80,
                           facecolors="none", edgecolors=[fcolor[f]] * len(nr),
                           linewidths=1.1, zorder=4)
                ax.scatter(nr["age"], [NR_LEVEL] * len(nr), marker="x", s=36,
                           color=fcolor[f], linewidths=1.1, zorder=4)
                flg = nr[nr["flagged"]]
                if len(flg):
                    ax.scatter(flg["age"], [NR_LEVEL] * len(flg), marker="s", s=120,
                               facecolors="none", edgecolors=flag_color,
                               linewidths=1.3, zorder=5)
        _db_yaxis(ax)
        ax.set_xlabel("Age at test (years)")
        ax.set_title(f"{ear} ear")

    freq_handles = [Line2D([0], [0], color=fcolor[f], lw=2.4, label=FREQ_LABEL[f] + " Hz")
                    for f in FREQ_ORDER]
    leg1 = axes[1].legend(handles=freq_handles, loc="lower right", fontsize=8,
                          title="Frequency", framealpha=0.9, ncol=2)
    axes[1].add_artist(leg1)
    shape_handles = [
        Line2D([0], [0], marker="o", color="grey", lw=0, label="Pure-tone"),
        Line2D([0], [0], marker="^", color="grey", lw=0, label="Play / behavioural"),
        Line2D([0], [0], marker="o", color="white", lw=0, markeredgecolor=flag_color,
               markeredgewidth=1.4, label="Conductive flag (verify)"),
        Line2D([0], [0], marker="s", color="grey", lw=0, markerfacecolor="none",
               markersize=9, label=f"No response ({NR_LEVEL} dB)"),
    ]
    axes[0].legend(handles=shape_handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Per-frequency threshold trajectories — participant {participant}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def figure_mega_waterfall(df, flag_color="#d62728"):
    """Pooled cohort: every audiogram from every participant, colour = age at test.
    EXPLORATORY. Repeated measures per participant are NOT independent; apparent
    age-grading is confounded by within-subject clustering and by which participants
    occupy which age range. Descriptive overview only."""
    ages = df["age"].dropna()
    amin, amax = ages.min(), ages.max()
    norm = plt.Normalize(amin, amax)
    # green -> teal -> blue -> purple -> magenta; avoids a washed-out midpoint
    cmap = LinearSegmentedColormap.from_list(
        "age_ref", ["#6cbf43", "#1aa07a", "#1f9bd0", "#5a6fc0", "#9b5fb8", "#e85aa0"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, ear in zip(axes, ["Right", "Left"]):
        de = df[df.ear == ear]
        for (pid, date, age), g in sorted(
                de.groupby(["participant", "date", "age"]), key=lambda kv: kv[0][2]):
            g = g.sort_values("freq_hz")
            color = cmap(norm(age))
            ax.plot(g["freq_hz"], g["threshold"], "-", color=color, alpha=0.5,
                    lw=0.9, zorder=2)
            pres = g[g["threshold"].notna()]
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"], s=10, color=color,
                           alpha=0.6, zorder=3, edgecolors="none")
            nr = g[g["no_response"]]
            if len(nr):
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="x", s=22,
                           color=color, alpha=0.6, linewidths=1.0, zorder=3)
        _audiogram_xaxis(ax); _db_yaxis(ax)
        ax.set_title(f"{ear} ear")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Age at test (years)")
    fig.suptitle(f"Pooled cohort audiograms — {df.participant.nunique()} participants, "
                 f"{df[['participant','date']].drop_duplicates().shape[0]} tests "
                 f"(EXPLORATORY; repeated measures not independent)", y=0.99, fontsize=11)
    return fig

def figure_grid_waterfall(df, ear, ncols=4, base_color="#1f4e79", flag_color="#d62728"):
    """Small-multiple grid: one audiogram-waterfall panel per participant, single ear.
    Participants ordered by age at first test. Opacity normalised WITHIN each panel
    (youngest->oldest for that person); opacity is therefore comparable within a panel
    but NOT across panels, because age ranges differ between participants."""
    order = df.groupby("participant")["age"].min().sort_values().index.tolist()
    n = len(order)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.1 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for i, pid in enumerate(order):
        ax = axes[i]
        de = df[(df.participant == pid) & (df.ear == ear)]
        ages = de["age"].dropna()
        amin, amax = ages.min(), ages.max()
        span = (amax - amin) or 1.0
        for (date, age), g in sorted(de.groupby(["date", "age"]), key=lambda kv: kv[0][1]):
            g = g.sort_values("freq_hz")
            a = ALPHA_FLOOR + (ALPHA_TOP - ALPHA_FLOOR) * (age - amin) / span
            ax.plot(g["freq_hz"], g["threshold"], "-", color=base_color, alpha=a, lw=0.8, zorder=2)
            pres = g[g["threshold"].notna()]
            is_pta = g["test_class"].iloc[0] == "PTA"
            flagged = bool(g["flagged"].iloc[0])
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"],
                           marker=("o" if is_pta else "^"), s=13, color=base_color,
                           alpha=a, edgecolors=(flag_color if flagged else "none"),
                           linewidths=(1.0 if flagged else 0), zorder=3)
            nr = g[g["no_response"]]
            if len(nr):
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="s", s=38,
                           facecolors="none", edgecolors=base_color, alpha=a,
                           linewidths=0.9, zorder=4)
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="x", s=16,
                           color=base_color, alpha=a, linewidths=0.9, zorder=4)
                if flagged:
                    ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="s", s=60,
                               facecolors="none", edgecolors=flag_color, alpha=a,
                               linewidths=1.1, zorder=5)
        _audiogram_xaxis(ax); _db_yaxis(ax)
        ax.set_title(f"{pid}  ({amin:.1f}\u2013{amax:.1f} yr)", fontsize=9)
        ax.label_outer()
    for j in range(n, len(axes)):
        axes[j].axis("off")
    handles = [
        Line2D([0], [0], marker="o", color=base_color, lw=0.8, label="Pure-tone audiometry"),
        Line2D([0], [0], marker="^", color=base_color, lw=0, label="Play / behavioural"),
        Line2D([0], [0], marker="o", color=base_color, lw=0, markeredgecolor=flag_color,
               markeredgewidth=1.2, label="Conductive flag (verify)"),
        Line2D([0], [0], marker="s", color=base_color, lw=0, markerfacecolor="none",
               markersize=9, label=f"No response ({NR_LEVEL} dB)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Serial audiograms by participant \u2014 {ear} ear "
                 f"(opacity \u221d age within each panel; youngest\u2192oldest)",
                 y=0.997, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def figure_grid_trajectories(df, ear, ncols=4, flag_color="#d62728"):
    """Small-multiple grid: one threshold-vs-age panel per participant, single ear.
    Lines coloured by frequency. Participants ordered by age at first test.
    Shared axes across panels for comparability."""
    order = df.groupby("participant")["age"].min().sort_values().index.tolist()
    n = len(order)
    nrows = int(np.ceil(n / ncols))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(FREQ_ORDER)))
    fcolor = {f: cmap[i] for i, f in enumerate(FREQ_ORDER)}
    # shared age axis across panels for comparability
    amin_all, amax_all = df["age"].min(), df["age"].max()
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.1 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for i, pid in enumerate(order):
        ax = axes[i]
        de = df[(df.participant == pid) & (df.ear == ear)]
        for f in FREQ_ORDER:
            gf = de[de.freq_hz == f].sort_values("age")
            ax.plot(gf["age"], gf["threshold"], "-", color=fcolor[f], lw=0.9,
                    alpha=0.9, zorder=2)
            pres = gf[gf["threshold"].notna()]
            for cls, mk in [("PTA", "o"), ("Other", "^")]:
                sub = pres[pres.test_class == cls]
                if not len(sub):
                    continue
                edge = [flag_color if fl else fcolor[f] for fl in sub["flagged"]]
                lw = [1.2 if fl else 0.4 for fl in sub["flagged"]]
                ax.scatter(sub["age"], sub["threshold"], marker=mk, s=14,
                           facecolors=[fcolor[f]] * len(sub), edgecolors=edge,
                           linewidths=lw, zorder=3)
            nr = gf[gf["no_response"]]
            if len(nr):
                ax.scatter(nr["age"], [NR_LEVEL] * len(nr), marker="x", s=18,
                           color=fcolor[f], linewidths=1.0, zorder=4)
        _db_yaxis(ax)
        ax.set_xlim(max(0, amin_all - 1), amax_all + 1)
        ax.set_xlabel("Age (years)")
        ax.set_title(f"{pid}", fontsize=9)
        ax.label_outer()
    for j in range(n, len(axes)):
        axes[j].axis("off")
    freq_handles = [Line2D([0], [0], color=fcolor[f], lw=2.2, label=FREQ_LABEL[f] + " Hz")
                    for f in FREQ_ORDER]
    shape_handles = [
        Line2D([0], [0], marker="o", color="grey", lw=0, label="Pure-tone"),
        Line2D([0], [0], marker="^", color="grey", lw=0, label="Play / behavioural"),
        Line2D([0], [0], marker="o", color="white", lw=0, markeredgecolor=flag_color,
               markeredgewidth=1.2, label="Conductive flag (verify)"),
        Line2D([0], [0], marker="x", color="grey", lw=0, label=f"No response ({NR_LEVEL} dB)"),
    ]
    fig.legend(handles=freq_handles + shape_handles, loc="lower center",
               ncol=6, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Per-frequency threshold trajectories by participant \u2014 {ear} ear "
                 f"(colour = frequency; shared age axis)", y=0.997, fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig

def figure_grid_waterfall_paired(df, per_row=3, base_color="#1f4e79", flag_color="#d62728"):
    """Combined grid: each participant shown as an adjacent Right|Left pair, so the two
    ears sit side by side for interaural comparison. Participants ordered by age at first
    test. Opacity normalised within each participant's COMBINED (both-ear) age range, so
    the R and L panels of one person share the same age->opacity mapping."""
    order = df.groupby("participant")["age"].min().sort_values().index.tolist()
    n = len(order)
    nrows = int(np.ceil(n / per_row))
    # column layout: groups of [panel, panel] separated by a thin spacer column
    width = []
    for g in range(per_row):
        if g > 0:
            width.append(0.30)
        width += [1, 1]
    ncols = len(width)
    spacer_cols = {3 * g - 1 for g in range(1, per_row)}
    fig, axes = plt.subplots(nrows, ncols, figsize=(sum(width) * 2.6, 2.95 * nrows),
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

    def draw_ear(ax, de, amin, amax):
        span = (amax - amin) or 1.0
        for (date, age), gg in sorted(de.groupby(["date", "age"]), key=lambda kv: kv[0][1]):
            gg = gg.sort_values("freq_hz")
            a = ALPHA_FLOOR + (ALPHA_TOP - ALPHA_FLOOR) * (age - amin) / span
            ax.plot(gg["freq_hz"], gg["threshold"], "-", color=base_color, alpha=a,
                    lw=0.8, zorder=2)
            pres = gg[gg["threshold"].notna()]
            is_pta = gg["test_class"].iloc[0] == "PTA"
            flagged = bool(gg["flagged"].iloc[0])
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"],
                           marker=("o" if is_pta else "^"), s=12, color=base_color,
                           alpha=a, edgecolors=(flag_color if flagged else "none"),
                           linewidths=(0.9 if flagged else 0), zorder=3)
            nr = gg[gg["no_response"]]
            if len(nr):
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="s", s=34,
                           facecolors="none", edgecolors=base_color, alpha=a,
                           linewidths=0.8, zorder=4)
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="x", s=14,
                           color=base_color, alpha=a, linewidths=0.8, zorder=4)
                if flagged:
                    ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="s", s=54,
                               facecolors="none", edgecolors=flag_color, alpha=a,
                               linewidths=1.0, zorder=5)

    for i, pid in enumerate(order):
        row, g = i // per_row, i % per_row
        b = g * 3
        ages = df[df.participant == pid]["age"].dropna()
        amin, amax = ages.min(), ages.max()
        for c, ear, tag in [(b, "Right", "R"), (b + 1, "Left", "L")]:
            ax = axes[row, c]
            de = df[(df.participant == pid) & (df.ear == ear)]
            draw_ear(ax, de, amin, amax)
            _audiogram_xaxis(ax); _db_yaxis(ax)
            if tag == "R":
                ax.set_title(f"{pid}  R   {amin:.1f}\u2013{amax:.1f}y", fontsize=8.5)
            else:
                ax.set_title(f"{pid}  L", fontsize=8.5)
            if c != 0:
                ax.set_ylabel(""); ax.tick_params(labelleft=False)
            if row != col_bottom[c]:
                ax.set_xlabel(""); ax.tick_params(labelbottom=False)

    # turn off spacer columns and unused panels
    for r in range(nrows):
        for c in range(ncols):
            if c in spacer_cols or (r, c) not in used:
                axes[r, c].axis("off")

    handles = [
        Line2D([0], [0], marker="o", color=base_color, lw=0.8, label="Pure-tone audiometry"),
        Line2D([0], [0], marker="^", color=base_color, lw=0, label="Play / behavioural"),
        Line2D([0], [0], marker="o", color=base_color, lw=0, markeredgecolor=flag_color,
               markeredgewidth=1.2, label="Conductive flag (verify)"),
        Line2D([0], [0], marker="s", color=base_color, lw=0, markerfacecolor="none",
               markersize=9, label=f"No response ({NR_LEVEL} dB)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Serial audiograms by participant \u2014 Right | Left paired "
                 "(opacity \u221d age within participant; youngest\u2192oldest)",
                 y=0.997, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def figure_mega_waterfall_subset(df, participants, label=None, norm_range=None,
                                 flag_color="#d62728"):
    """Pooled waterfall for a subset of participants (e.g. one genotype group).
    Pass norm_range=(amin, amax) to share one age colour scale across several groups,
    otherwise each figure self-scales and the groups are NOT visually comparable.
    EXPLORATORY: small per-group n; apparent between-group differences are confounded
    with age structure and test count."""
    d = df[df.participant.isin(participants)]
    ages = d["age"].dropna()
    amin, amax = norm_range if norm_range is not None else (ages.min(), ages.max())
    norm = plt.Normalize(amin, amax)
    cmap = LinearSegmentedColormap.from_list(
        "age_ref", ["#6cbf43", "#1aa07a", "#1f9bd0", "#5a6fc0", "#9b5fb8", "#e85aa0"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, ear in zip(axes, ["Right", "Left"]):
        de = d[d.ear == ear]
        for (pid, date, age), g in sorted(
                de.groupby(["participant", "date", "age"]), key=lambda kv: kv[0][2]):
            g = g.sort_values("freq_hz")
            color = cmap(norm(age))
            ax.plot(g["freq_hz"], g["threshold"], "-", color=color, alpha=0.5,
                    lw=0.9, zorder=2)
            pres = g[g["threshold"].notna()]
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"], s=10, color=color,
                           alpha=0.6, zorder=3, edgecolors="none")
            nr = g[g["no_response"]]
            if len(nr):
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="x", s=22,
                           color=color, alpha=0.6, linewidths=1.0, zorder=3)
        _audiogram_xaxis(ax); _db_yaxis(ax)
        ax.set_title(f"{ear} ear")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Age at test (years)")
    head = (label + " — ") if label else ""
    fig.suptitle(f"{head}Pooled audiograms — {d.participant.nunique()} participants, "
                 f"{d[['participant','date']].drop_duplicates().shape[0]} tests "
                 f"(EXPLORATORY; repeated measures not independent)", y=0.99, fontsize=11)
    return fig

# ---- genotype-stratified pooled waterfall --------------------------------
# Group membership (e.g. genotype class) is read from an external JSON file rather
# than hardcoded, so that participant identifiers stay out of version control.
# See groups.example.json for the expected format:
#   {"Group label": ["ID1", "ID2", ...], "Another group": ["ID3", ...]}

def load_groups(path="groups.json"):
    """Read participant group membership from a JSON file."""
    import json
    with open(path) as f:
        return json.load(f)


def figure_mega_waterfall_subset(df, participants, label=None, norm_range=None):
    """Pooled waterfall restricted to a subset of participants (e.g. one genotype class).

    Pass norm_range=(amin, amax) to share a single age colour scale across several
    groups; without it each figure self-scales and the groups are NOT comparable.

    EXPLORATORY. Per-group n is small, repeated measures within a participant are not
    independent, and groups typically differ in age structure and number of tests. Any
    apparent between-group difference is confounded with those factors. Descriptive only.
    """
    d = df[df.participant.isin(participants)]
    if d.empty:
        raise ValueError("No matching participants for this group.")
    ages = d["age"].dropna()
    amin, amax = norm_range if norm_range is not None else (ages.min(), ages.max())
    norm = plt.Normalize(amin, amax)
    cmap = LinearSegmentedColormap.from_list(
        "age_ref", ["#6cbf43", "#1aa07a", "#1f9bd0", "#5a6fc0", "#9b5fb8", "#e85aa0"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, ear in zip(axes, ["Right", "Left"]):
        de = d[d.ear == ear]
        for (pid, date, age), g in sorted(
                de.groupby(["participant", "date", "age"]), key=lambda kv: kv[0][2]):
            g = g.sort_values("freq_hz")
            color = cmap(norm(age))
            ax.plot(g["freq_hz"], g["threshold"], "-", color=color, alpha=0.5,
                    lw=0.9, zorder=2)
            pres = g[g["threshold"].notna()]
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"], s=10, color=color,
                           alpha=0.6, zorder=3, edgecolors="none")
            nr = g[g["no_response"]]
            if len(nr):
                ax.scatter(nr["freq_hz"], [NR_LEVEL] * len(nr), marker="x", s=22,
                           color=color, alpha=0.6, linewidths=1.0, zorder=3)
        _audiogram_xaxis(ax); _db_yaxis(ax)
        ax.set_title(f"{ear} ear")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Age at test (years)")
    head = (label + " \u2014 ") if label else ""
    fig.suptitle(f"{head}Pooled audiograms \u2014 {d.participant.nunique()} participants, "
                 f"{d[['participant','date']].drop_duplicates().shape[0]} tests "
                 f"(EXPLORATORY; repeated measures not independent)", y=0.99, fontsize=11)
    return fig


def figures_by_group(df, groups, out_dir=".", prefix="figG_mega"):
    """Generate one pooled waterfall per group, on a SHARED age colour scale.

    groups: dict of {label: [participant ids]}, e.g. from load_groups().
    Returns (summary DataFrame, list of unassigned participants) so that the n per
    group and any participants excluded from the grouping are reported explicitly.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    members = [p for v in groups.values() for p in v]
    sub = df[df.participant.isin(members)]
    if sub.empty:
        raise ValueError("None of the grouped participants are present in the data.")
    rng = (sub["age"].min(), sub["age"].max())   # shared scale => groups comparable
    rows = []
    for i, (label, ids) in enumerate(groups.items()):
        fig = figure_mega_waterfall_subset(df, ids, label, rng)
        safe = "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()
        fig.savefig(f"{out_dir}/{prefix}_{i+1}_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        d = df[df.participant.isin(ids)]
        rows.append({"group": label,
                     "participants": d.participant.nunique(),
                     "tests": d[["participant", "date"]].drop_duplicates().shape[0],
                     "age_min": round(d["age"].min(), 1),
                     "age_max": round(d["age"].max(), 1)})
    unassigned = sorted(set(df.participant.unique()) - set(members))
    return pd.DataFrame(rows), unassigned


if __name__ == "__main__":
    import os
    import sys
    # Input workbook path may be given as a command-line argument:
    #     python audiometry.py path/to/workbook.xlsx
    PATH = sys.argv[1] if len(sys.argv) > 1 else "audiometry_data.xlsx"
    OUT = "figs"
    os.makedirs(OUT, exist_ok=True)
    df = load_workbook(PATH)
    df.to_csv("long_format.csv", index=False)
    flag_table(df).to_csv("flag_verification.csv", index=False)
    for p in df.participant.unique():
        figure_waterfall(df, p).savefig(f"{OUT}/figA_waterfall_{p}.png", dpi=140, bbox_inches="tight")
        figure_trajectories(df, p).savefig(f"{OUT}/figB_trajectories_{p}.png", dpi=140, bbox_inches="tight")
        plt.close("all")
    figure_mega_waterfall(df).savefig(f"{OUT}/figC_mega_waterfall_cohort.png",
                                      dpi=150, bbox_inches="tight")
    plt.close("all")
    for ear in ["Right", "Left"]:
        figure_grid_waterfall(df, ear).savefig(
            f"{OUT}/figD_grid_waterfall_{ear.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        figure_grid_trajectories(df, ear).savefig(
            f"{OUT}/figE_grid_trajectories_{ear.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    figure_grid_waterfall_paired(df).savefig(
        f"{OUT}/figF_grid_waterfall_paired.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    # summary
    g = df.groupby("participant")
    summ = pd.DataFrame({
        "tests": g.apply(lambda d: d[["date"]].drop_duplicates().shape[0], include_groups=False),
        "age_min": g["age"].min().round(1),
        "age_max": g["age"].max().round(1),
        "no_response": g["no_response"].sum().astype(int),
        "flagged_tests": g.apply(lambda d: d.loc[d.flagged, "date"].nunique(), include_groups=False),
    })
    print(summ.to_string())

    # Optional: genotype- (or other-) stratified pooled waterfalls.
    # Runs only if a local groups.json is present; see groups.example.json.
    if os.path.exists("groups.json"):
        groups = load_groups("groups.json")
        gsumm, unassigned = figures_by_group(df, groups, out_dir=OUT)
        print("\nGroup composition:")
        print(gsumm.to_string(index=False))
        print("Unassigned participants (e.g. group not determined):", unassigned)
    print("\nTotal rows:", len(df), "| total no-response obs:", int(df.no_response.sum()))