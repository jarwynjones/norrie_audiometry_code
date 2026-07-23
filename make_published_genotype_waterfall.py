import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import audiometry as A
import make_published_grid2 as M

df = M.df

GROUPS = {
    "Truncating + large deletions": [
        "Exon 2 del ND44", "Exon 2+3 del Gong II 1", "Exon 2+3 del Gong III 1",
        "Intragenic del exon 2 Halpin", "c.325C>T ND31", "c.325C>T ND32",
        "c.49delG II 3", "c.49delG II 5", "c.49delG III 1"],
    "Non-truncating (missense)": [
        "c.277T>C fam member 6", "c.362G>A ND38", "p.Leu61Pro Halpin"],
    "Unknown genotype": [
        "Parving 1977 no mutation", "Parving 1978 no mutation", "Skevas no mutation"],
}
ORDER = ["Non-truncating (missense)", "Truncating + large deletions", "Unknown genotype"]

# same colourmap as the previous waterfall (green -> teal -> blue -> purple -> magenta)
CMAP = LinearSegmentedColormap.from_list(
    "age_ref", ["#6cbf43", "#1aa07a", "#1f9bd0", "#5a6fc0", "#9b5fb8", "#e85aa0"])

def figure_genotype_waterfall(df, ear_mode="both"):
    amin, amax = df["age"].min(), df["age"].max()   # shared scale across panels
    norm = plt.Normalize(amin, amax)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), sharex=True, sharey=True)
    for ax, gname in zip(axes, ORDER):
        members = GROUPS[gname]
        d = df[df.participant.isin(members)]
        for (pid, age, ear), g in sorted(d.groupby(["participant", "age", "ear"]),
                                         key=lambda kv: kv[0][1]):
            g = g.sort_values("freq_hz")
            color = CMAP(norm(age))
            ax.plot(g["freq_hz"], g["threshold"], "-", color=color, alpha=0.55,
                    lw=1.0, zorder=2)
            pres = g[g["threshold"].notna()]
            if len(pres):
                ax.scatter(pres["freq_hz"], pres["threshold"], s=15, color=color,
                           alpha=0.7, edgecolors="none", zorder=3)
            nr = g[g["nr_level"].notna()]
            for _, rr in nr.iterrows():
                x, y = rr["freq_hz"], rr["nr_level"]
                ax.scatter([x], [y], marker="o", s=24, facecolors="none",
                           edgecolors=color, linewidths=1.2, zorder=5)
                ax.annotate("", xy=(x, y + 11), xytext=(x, y + 1.5),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2), zorder=5)
        A._audiogram_xaxis(ax); A._db_yaxis(ax)
        n_case = d.participant.nunique()
        n_aud = d[["participant", "age"]].drop_duplicates().shape[0]
        ax.set_title(f"{gname}\n{n_case} cases, {n_aud} audiograms", fontsize=10.5,
                     linespacing=1.4)
        ax.label_outer()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.015)
    cbar.set_label("Age at test (years)")
    handles = [
        Line2D([0], [0], marker="o", color="grey", lw=1.0, label="Measured threshold"),
        Line2D([0], [0], marker="o", color="grey", lw=0, markerfacecolor="none",
               markeredgecolor="grey", label="No response (arrow = \u2265 level)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Published Norrie disease audiograms by genotype class "
                 "(both ears pooled; colour = age at test)", y=1.0, fontsize=12)
    return fig

fig = figure_genotype_waterfall(df)
fig.savefig("figN_genotype_waterfall.png", dpi=150, bbox_inches="tight")
plt.close("all")

# report composition
for gname in ORDER:
    d = df[df.participant.isin(GROUPS[gname])]
    print(f"{gname:32s} cases={d.participant.nunique():2d}  "
          f"audiograms={d[['participant','age']].drop_duplicates().shape[0]:2d}  "
          f"ages={sorted(d.age.unique())}")
assigned = {p for v in GROUPS.values() for p in v}
print("Unassigned:", sorted(set(df.participant.unique()) - assigned))
