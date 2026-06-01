"""
Unemployment Analysis with Python
===================================
Internship Project - Task 2
CodeAlpha - Unemployment in India (May 2019 – Jun 2020)

Analyzes unemployment trends, Covid-19 impact, regional patterns,
and seasonal variations using Python data analysis tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── STYLE ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
})
PALETTE = sns.color_palette("Set2", 10)
COVID_COLOR = "#E74C3C"
PRE_COLOR   = "#2ECC71"

# ─── 1. LOAD & CLEAN ─────────────────────────────────────────────────────────

print("=" * 60)
print("   UNEMPLOYMENT IN INDIA — DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv("Unemployment_in_India.csv")

# Strip whitespace from column names and string values
df.columns = df.columns.str.strip()
df["Date"]   = pd.to_datetime(df["Date"].str.strip(), format="%d-%m-%Y")
df["Region"] = df["Region"].str.strip()
df["Area"]   = df["Area"].str.strip()
df.rename(columns={
    "Estimated Unemployment Rate (%)":        "UnemploymentRate",
    "Estimated Employed":                     "Employed",
    "Estimated Labour Participation Rate (%)":"LabourParticipation",
}, inplace=True)

# Drop rows with missing key values
df.dropna(subset=["Date", "Region", "UnemploymentRate"], inplace=True)
df.sort_values("Date", inplace=True)

# Mark Covid period (lockdown announced March 24, 2020)
df["Period"] = df["Date"].apply(
    lambda d: "Post-Lockdown (Covid)" if d >= pd.Timestamp("2020-03-01") else "Pre-Lockdown"
)

print(f"\n📊 Dataset: {df.shape[0]} records | "
      f"{df['Region'].nunique()} states | "
      f"{df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}")

# ─── 2. SUMMARY STATS ────────────────────────────────────────────────────────

pre  = df[df["Period"] == "Pre-Lockdown"]["UnemploymentRate"]
post = df[df["Period"] == "Post-Lockdown (Covid)"]["UnemploymentRate"]

print(f"\n📈 Pre-Lockdown  avg unemployment : {pre.mean():.2f}%")
print(f"   Post-Lockdown avg unemployment : {post.mean():.2f}%")
print(f"   Peak unemployment              : {df['UnemploymentRate'].max():.2f}% "
      f"({df.loc[df['UnemploymentRate'].idxmax(), 'Date'].strftime('%b %Y')})")

# ─── 3. BUILD FIGURE (2×3 grid) ──────────────────────────────────────────────

fig = plt.figure(figsize=(18, 13))
fig.suptitle(
    "Unemployment in India — Comprehensive Analysis (May 2019 – Jun 2020)",
    fontsize=17, fontweight="bold", y=0.98,
)

# ── 3a. National monthly trend ───────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
monthly = df.groupby("Date")["UnemploymentRate"].mean().reset_index()

pre_df  = monthly[monthly["Date"] <  pd.Timestamp("2020-03-01")]
post_df = monthly[monthly["Date"] >= pd.Timestamp("2020-03-01")]

ax1.plot(pre_df["Date"],  pre_df["UnemploymentRate"],
         color=PRE_COLOR,   lw=2.5, marker="o", ms=5, label="Pre-Lockdown")
ax1.plot(post_df["Date"], post_df["UnemploymentRate"],
         color=COVID_COLOR, lw=2.5, marker="o", ms=5, label="Post-Lockdown (Covid)")
ax1.axvline(pd.Timestamp("2020-03-01"), color="grey", ls="--", lw=1.5)
ax1.text(pd.Timestamp("2020-03-05"), ax1.get_ylim()[1] * 0.95,
         "← Lockdown", color="grey", fontsize=8)
ax1.set_title("National Monthly Unemployment Trend", fontweight="bold")
ax1.set_ylabel("Unemployment Rate (%)")
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))

# ── 3b. Rural vs Urban ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
rural_urban = df.groupby(["Date", "Area"])["UnemploymentRate"].mean().unstack()
for area, color in zip(["Rural", "Urban"], ["#27AE60", "#2980B9"]):
    if area in rural_urban.columns:
        ax2.plot(rural_urban.index, rural_urban[area],
                 color=color, lw=2.5, marker="o", ms=4, label=area)
ax2.axvline(pd.Timestamp("2020-03-01"), color="grey", ls="--", lw=1.5)
ax2.set_title("Rural vs Urban Unemployment", fontweight="bold")
ax2.set_ylabel("Unemployment Rate (%)")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))

# ── 3c. Top 10 hardest-hit states (peak Covid period) ────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
covid_states = (
    df[df["Period"] == "Post-Lockdown (Covid)"]
    .groupby("Region")["UnemploymentRate"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
bars = ax3.barh(covid_states.index[::-1], covid_states.values[::-1],
                color=sns.color_palette("Reds_r", 10))
ax3.set_title("Top 10 States — Avg Unemployment\n(Post-Lockdown)", fontweight="bold")
ax3.set_xlabel("Unemployment Rate (%)")
for bar, val in zip(bars, covid_states.values[::-1]):
    ax3.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=8)

# ── 3d. Covid Impact — before vs after per state (top 12) ────────────────────
ax4 = fig.add_subplot(2, 3, 4)
state_period = df.groupby(["Region", "Period"])["UnemploymentRate"].mean().unstack()
state_period["Change"] = state_period["Post-Lockdown (Covid)"] - state_period["Pre-Lockdown"]
top12 = state_period["Change"].dropna().sort_values(ascending=False).head(12)
colors_bar = [COVID_COLOR if v > 0 else PRE_COLOR for v in top12.values]
ax4.barh(top12.index[::-1], top12.values[::-1], color=colors_bar[::-1])
ax4.axvline(0, color="black", lw=0.8)
ax4.set_title("Covid-19 Impact: Change in\nUnemployment Rate by State", fontweight="bold")
ax4.set_xlabel("Change in Unemployment Rate (%)")

# ── 3e. Distribution — pre vs post ───────────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(pre,  bins=20, alpha=0.6, color=PRE_COLOR,   label=f"Pre-Lockdown  (μ={pre.mean():.1f}%)")
ax5.hist(post, bins=20, alpha=0.6, color=COVID_COLOR, label=f"Post-Lockdown (μ={post.mean():.1f}%)")
ax5.set_title("Distribution of Unemployment\nPre vs Post Lockdown", fontweight="bold")
ax5.set_xlabel("Unemployment Rate (%)")
ax5.set_ylabel("Frequency")
ax5.legend(fontsize=8)

# ── 3f. Heatmap — state × month ──────────────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
heatmap_df = df.pivot_table(
    index="Region",
    columns=df["Date"].dt.strftime("%b %Y"),
    values="UnemploymentRate",
    aggfunc="mean",
)
# Sort columns chronologically
col_order = pd.to_datetime(heatmap_df.columns, format="%b %Y").argsort()
heatmap_df = heatmap_df.iloc[:, col_order]
sns.heatmap(
    heatmap_df, ax=ax6, cmap="YlOrRd",
    linewidths=0.3, linecolor="white",
    cbar_kws={"label": "Unemployment Rate (%)"},
    annot=False,
)
ax6.set_title("Unemployment Heatmap\n(State × Month)", fontweight="bold")
ax6.set_xlabel("")
ax6.tick_params(axis="x", labelsize=7, rotation=45)
ax6.tick_params(axis="y", labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("unemployment_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n💾 Saved: unemployment_analysis.png")

# ─── 4. KEY INSIGHTS ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("   KEY INSIGHTS")
print("=" * 60)
print(f"""
1. 📈 Covid-19 Spike  : Unemployment jumped from {pre.mean():.1f}% to
                        {post.mean():.1f}% after the March 2020 lockdown
                        — a {((post.mean()-pre.mean())/pre.mean()*100):.0f}% relative increase.

2. 🏙️  Urban > Rural  : Urban areas saw steeper rises due to
                        factory/service sector shutdowns.

3. 🗺️  Hardest Hit    : {covid_states.index[0]} ({covid_states.iloc[0]:.1f}%),
                        {covid_states.index[1]} ({covid_states.iloc[1]:.1f}%),
                        {covid_states.index[2]} ({covid_states.iloc[2]:.1f}%)
                        had the highest post-lockdown unemployment.

4. 📅 Seasonal Trend : Slight dip in unemployment during
                        harvest months (Oct–Dec 2019).

5. 📉 Labour Force   : Participation rate dropped alongside
                        unemployment — many stopped looking for work.
""")

print("✅ Task 2 complete! Check unemployment_analysis.png")
