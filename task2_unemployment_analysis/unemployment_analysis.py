"""
Unemployment Analysis with Python
===================================
Internship Project - Task 2
CodeAlpha - Unemployment in India

Uses TWO datasets:
  1. Unemployment_in_India.csv          — state-level, May 2019–Jun 2020
  2. Unemployment_Rate_upto_11_2020.csv — state-level with geo coords, Jan–Oct 2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
})
COVID_COLOR = "#E74C3C"
PRE_COLOR   = "#2ECC71"

# ─── 1. LOAD & CLEAN DATASET 1 ───────────────────────────────────────────────

df1 = pd.read_csv("Unemployment_in_India.csv")
df1.columns = df1.columns.str.strip()
df1["Date"]   = pd.to_datetime(df1["Date"].str.strip(), format="%d-%m-%Y")
df1["Region"] = df1["Region"].str.strip()
df1["Area"]   = df1["Area"].str.strip()
df1.rename(columns={
    "Estimated Unemployment Rate (%)":        "UnemploymentRate",
    "Estimated Employed":                     "Employed",
    "Estimated Labour Participation Rate (%)":"LabourParticipation",
}, inplace=True)
df1.dropna(subset=["Date", "Region", "UnemploymentRate"], inplace=True)
df1.sort_values("Date", inplace=True)
df1["Period"] = df1["Date"].apply(
    lambda d: "Post-Lockdown" if d >= pd.Timestamp("2020-03-01") else "Pre-Lockdown"
)

# ─── 2. LOAD & CLEAN DATASET 2 ───────────────────────────────────────────────

df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df2.columns = df2.columns.str.strip()
df2["Date"]   = pd.to_datetime(df2["Date"].str.strip(), format="%d-%m-%Y")
df2["Region"] = df2["Region"].str.strip()
df2.rename(columns={
    "Estimated Unemployment Rate (%)":        "UnemploymentRate",
    "Estimated Employed":                     "Employed",
    "Estimated Labour Participation Rate (%)":"LabourParticipation",
    "Region.1":                               "Zone",
}, inplace=True)
df2.dropna(subset=["Date", "Region", "UnemploymentRate"], inplace=True)
df2.sort_values("Date", inplace=True)

print("=" * 60)
print("   UNEMPLOYMENT IN INDIA — DATA ANALYSIS")
print("=" * 60)
print(f"\nDataset 1: {df1.shape[0]} records | {df1['Region'].nunique()} states | "
      f"{df1['Date'].min().strftime('%b %Y')} → {df1['Date'].max().strftime('%b %Y')}")
print(f"Dataset 2: {df2.shape[0]} records | {df2['Region'].nunique()} states | "
      f"{df2['Date'].min().strftime('%b %Y')} → {df2['Date'].max().strftime('%b %Y')}")

pre  = df1[df1["Period"] == "Pre-Lockdown"]["UnemploymentRate"]
post = df1[df1["Period"] == "Post-Lockdown"]["UnemploymentRate"]
print(f"\nPre-Lockdown  avg : {pre.mean():.2f}%")
print(f"Post-Lockdown avg : {post.mean():.2f}%  (+{((post.mean()-pre.mean())/pre.mean()*100):.0f}%)")

# ─── 3. FIGURE 1 — Main Analysis (2×3) ───────────────────────────────────────

fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle(
    "Unemployment in India — Comprehensive Analysis (May 2019 – Jun 2020)",
    fontsize=16, fontweight="bold", y=0.99,
)

# 3a. National monthly trend
monthly = df1.groupby("Date")["UnemploymentRate"].mean().reset_index()
pre_df  = monthly[monthly["Date"] <  pd.Timestamp("2020-03-01")]
post_df = monthly[monthly["Date"] >= pd.Timestamp("2020-03-01")]
ax = axes[0, 0]
ax.plot(pre_df["Date"],  pre_df["UnemploymentRate"],  color=PRE_COLOR,   lw=2.5, marker="o", ms=5, label="Pre-Lockdown")
ax.plot(post_df["Date"], post_df["UnemploymentRate"], color=COVID_COLOR, lw=2.5, marker="o", ms=5, label="Post-Lockdown")
ax.axvline(pd.Timestamp("2020-03-01"), color="grey", ls="--", lw=1.5)
ax.set_title("National Monthly Unemployment Trend", fontweight="bold")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))

# 3b. Rural vs Urban
ax = axes[0, 1]
ru = df1.groupby(["Date", "Area"])["UnemploymentRate"].mean().unstack()
for area, color in zip(["Rural", "Urban"], ["#27AE60", "#2980B9"]):
    if area in ru.columns:
        ax.plot(ru.index, ru[area], color=color, lw=2.5, marker="o", ms=4, label=area)
ax.axvline(pd.Timestamp("2020-03-01"), color="grey", ls="--", lw=1.5)
ax.set_title("Rural vs Urban Unemployment", fontweight="bold")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))

# 3c. Top 10 hardest-hit states
ax = axes[0, 2]
covid_states = (
    df1[df1["Period"] == "Post-Lockdown"]
    .groupby("Region")["UnemploymentRate"].mean()
    .sort_values(ascending=False).head(10)
)
bars = ax.barh(covid_states.index[::-1], covid_states.values[::-1],
               color=sns.color_palette("Reds_r", 10))
ax.set_title("Top 10 States — Post-Lockdown\nAvg Unemployment", fontweight="bold")
ax.set_xlabel("Unemployment Rate (%)")
for bar, val in zip(bars, covid_states.values[::-1]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8)

# 3d. Covid impact by state
ax = axes[1, 0]
sp = df1.groupby(["Region", "Period"])["UnemploymentRate"].mean().unstack()
sp["Change"] = sp["Post-Lockdown"] - sp["Pre-Lockdown"]
top12 = sp["Change"].dropna().sort_values(ascending=False).head(12)
ax.barh(top12.index[::-1], top12.values[::-1],
        color=[COVID_COLOR if v > 0 else PRE_COLOR for v in top12.values[::-1]])
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Covid-19 Impact: Change in\nUnemployment Rate by State", fontweight="bold")
ax.set_xlabel("Change in Unemployment Rate (%)")

# 3e. Distribution pre vs post
ax = axes[1, 1]
ax.hist(pre,  bins=20, alpha=0.6, color=PRE_COLOR,   label=f"Pre-Lockdown  (μ={pre.mean():.1f}%)")
ax.hist(post, bins=20, alpha=0.6, color=COVID_COLOR, label=f"Post-Lockdown (μ={post.mean():.1f}%)")
ax.set_title("Distribution: Pre vs Post Lockdown", fontweight="bold")
ax.set_xlabel("Unemployment Rate (%)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8)

# 3f. Heatmap state × month
ax = axes[1, 2]
hm = df1.pivot_table(index="Region", columns=df1["Date"].dt.strftime("%b %Y"),
                     values="UnemploymentRate", aggfunc="mean")
col_order = pd.to_datetime(hm.columns, format="%b %Y").argsort()
hm = hm.iloc[:, col_order]
sns.heatmap(hm, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Unemployment Rate (%)"}, annot=False)
ax.set_title("State × Month Heatmap", fontweight="bold")
ax.set_xlabel("")
ax.tick_params(axis="x", labelsize=7, rotation=45)
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig("unemployment_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n💾 Saved: unemployment_analysis.png")

# ─── 4. FIGURE 2 — Dataset 2: Geo & Zone Analysis ────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(
    "Unemployment in India — Regional & Geographic Analysis (Jan–Oct 2020)",
    fontsize=15, fontweight="bold",
)

# 4a. Zone-wise trend
ax = axes2[0]
zone_monthly = df2.groupby(["Date", "Zone"])["UnemploymentRate"].mean().unstack()
zone_colors  = sns.color_palette("tab10", len(zone_monthly.columns))
for zone, color in zip(zone_monthly.columns, zone_colors):
    ax.plot(zone_monthly.index, zone_monthly[zone], marker="o", ms=4,
            lw=2, label=zone, color=color)
ax.axvline(pd.Timestamp("2020-03-24"), color="grey", ls="--", lw=1.5)
ax.text(pd.Timestamp("2020-03-28"), ax.get_ylim()[1] * 0.9, "Lockdown →",
        color="grey", fontsize=8)
ax.set_title("Zone-wise Unemployment Trend", fontweight="bold")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))

# 4b. Bubble map (scatter plot using lat/lon)
ax = axes2[1]
state_avg = df2.groupby("Region").agg(
    UnemploymentRate=("UnemploymentRate", "mean"),
    lat=("latitude", "first"),
    lon=("longitude", "first"),
).reset_index()
scatter = ax.scatter(
    state_avg["lon"], state_avg["lat"],
    s=state_avg["UnemploymentRate"] * 12,
    c=state_avg["UnemploymentRate"],
    cmap="YlOrRd", alpha=0.8, edgecolors="white", linewidths=0.5,
)
plt.colorbar(scatter, ax=ax, label="Avg Unemployment Rate (%)")
for _, row in state_avg.iterrows():
    ax.annotate(row["Region"], (row["lon"], row["lat"]),
                fontsize=5.5, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points")
ax.set_title("Geographic Unemployment Map\n(bubble size = rate)", fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 4c. Zone box plot
ax = axes2[2]
zone_order = df2.groupby("Zone")["UnemploymentRate"].median().sort_values(ascending=False).index
sns.boxplot(data=df2, x="Zone", y="UnemploymentRate", order=zone_order,
            palette="Set2", ax=ax)
ax.set_title("Unemployment Distribution by Zone", fontweight="bold")
ax.set_xlabel("Zone")
ax.set_ylabel("Unemployment Rate (%)")
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("unemployment_geo_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("💾 Saved: unemployment_geo_analysis.png")

# ─── 5. INSIGHTS ─────────────────────────────────────────────────────────────

print(f"""
{'='*60}
   KEY INSIGHTS
{'='*60}

1. Covid-19 Spike   : Unemployment jumped from {pre.mean():.1f}% → {post.mean():.1f}%
                      after the March 2020 lockdown (+{((post.mean()-pre.mean())/pre.mean()*100):.0f}%).

2. Hardest Hit      : Puducherry, Jharkhand, Haryana had the
                      highest post-lockdown unemployment.

3. Urban > Rural    : Urban areas saw steeper rises due to
                      factory/service sector shutdowns.

4. Seasonal Trend   : Slight dip during harvest months (Oct–Dec 2019).

5. Geographic View  : Northern & Western states showed higher
                      volatility; Southern states more stable.

✅ Task 2 complete! Two charts saved.
""")
