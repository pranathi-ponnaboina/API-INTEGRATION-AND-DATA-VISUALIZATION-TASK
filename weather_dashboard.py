"""
============================================================
  CODTECH INTERNSHIP TASK — API Integration & Data Visualization
  Tool     : OpenWeatherMap API  (free tier)
  Fallback : Realistic synthetic data when network is unavailable
  Output   : weather_dashboard.png  (multi-panel dashboard)
============================================================
"""

# ── Standard imports ──────────────────────────────────────
import sys
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ══════════════════════════════════════════════════════════
# 1.  API INTEGRATION — OpenWeatherMap
# ══════════════════════════════════════════════════════════

API_KEY  = "YOUR_OPENWEATHERMAP_API_KEY"   # ← replace with your free key
CITY     = "Hyderabad"
COUNTRY  = "IN"
UNITS    = "metric"                         # Celsius

BASE_URL          = "https://api.openweathermap.org/data/2.5"
CURRENT_URL       = f"{BASE_URL}/weather"
FORECAST_URL      = f"{BASE_URL}/forecast"
AIR_QUALITY_URL   = f"{BASE_URL}/air_pollution/forecast"

def fetch_current_weather():
    """Fetch current weather from OpenWeatherMap."""
    params = {"q": f"{CITY},{COUNTRY}", "appid": API_KEY, "units": UNITS}
    r = requests.get(CURRENT_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_5day_forecast():
    """Fetch 5-day / 3-hour forecast from OpenWeatherMap."""
    params = {"q": f"{CITY},{COUNTRY}", "appid": API_KEY, "units": UNITS, "cnt": 40}
    r = requests.get(FORECAST_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def parse_forecast(raw):
    """Convert raw forecast JSON → tidy DataFrame."""
    rows = []
    for item in raw["list"]:
        rows.append({
            "datetime"    : datetime.datetime.fromtimestamp(item["dt"]),
            "temp"        : item["main"]["temp"],
            "feels_like"  : item["main"]["feels_like"],
            "humidity"    : item["main"]["humidity"],
            "wind_speed"  : item["wind"]["speed"],
            "description" : item["weather"][0]["description"].title(),
            "rain_mm"     : item.get("rain", {}).get("3h", 0.0),
            "cloud_pct"   : item["clouds"]["all"],
        })
    return pd.DataFrame(rows)

# ── Try live API; fall back to realistic synthetic data ───
def get_weather_data():
    """Return (current_dict, forecast_df) from live API or synthetic data."""
    if REQUESTS_AVAILABLE and API_KEY != "YOUR_OPENWEATHERMAP_API_KEY":
        try:
            print("📡  Connecting to OpenWeatherMap API …")
            current_raw  = fetch_current_weather()
            forecast_raw = fetch_5day_forecast()
            df = parse_forecast(forecast_raw)
            current = {
                "temp"       : current_raw["main"]["temp"],
                "feels_like" : current_raw["main"]["feels_like"],
                "humidity"   : current_raw["main"]["humidity"],
                "wind_speed" : current_raw["wind"]["speed"],
                "description": current_raw["weather"][0]["description"].title(),
                "city"       : current_raw["name"],
            }
            print("✅  Live data fetched successfully.")
            return current, df
        except Exception as e:
            print(f"⚠️  API error ({e}). Using synthetic data.")

    # ── Synthetic data (realistic Hyderabad May weather) ──
    print("🔄  Generating realistic synthetic weather data for Hyderabad …")
    np.random.seed(42)
    n = 40                          # 5 days × 8 readings/day
    base = datetime.datetime(2025, 5, 28, 0, 0)
    times = [base + datetime.timedelta(hours=3*i) for i in range(n)]

    # Diurnal temperature cycle + random noise
    hour_of_day   = np.array([t.hour for t in times])
    temp_base     = 32 + 6 * np.sin((hour_of_day - 6) * np.pi / 12)
    temp          = temp_base + np.random.normal(0, 0.8, n)
    feels_like    = temp - 2 + np.random.normal(0, 0.5, n)
    humidity      = 55 + 20 * np.cos((hour_of_day - 14) * np.pi / 12) + np.random.normal(0, 3, n)
    humidity      = np.clip(humidity, 30, 95)
    wind_speed    = np.abs(np.random.normal(3.5, 1.2, n))
    rain_chance   = np.random.choice([0]*7 + [0.5, 1.2, 2.5], size=n)
    cloud_pct     = np.random.randint(10, 80, n)
    descriptions  = np.random.choice(
        ["Clear Sky", "Partly Cloudy", "Haze", "Thunderstorm", "Light Rain"],
        size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05]
    )

    df = pd.DataFrame({
        "datetime"   : times,
        "temp"       : temp,
        "feels_like" : feels_like,
        "humidity"   : humidity,
        "wind_speed" : wind_speed,
        "rain_mm"    : rain_chance,
        "cloud_pct"  : cloud_pct,
        "description": descriptions,
    })

    current = {
        "temp"       : float(temp[0]),
        "feels_like" : float(feels_like[0]),
        "humidity"   : float(humidity[0]),
        "wind_speed" : float(wind_speed[0]),
        "description": descriptions[0],
        "city"       : CITY,
    }
    return current, df


# ══════════════════════════════════════════════════════════
# 2.  DATA PROCESSING
# ══════════════════════════════════════════════════════════

def process(df):
    """Derive extra columns used in plots."""
    df = df.copy()
    df["date"]         = df["datetime"].dt.date
    df["hour"]         = df["datetime"].dt.hour
    df["day_label"]    = df["datetime"].dt.strftime("%b %d")

    # Daily aggregates
    daily = df.groupby("date").agg(
        temp_max   = ("temp",       "max"),
        temp_min   = ("temp",       "min"),
        temp_mean  = ("temp",       "mean"),
        humidity   = ("humidity",   "mean"),
        wind_speed = ("wind_speed", "mean"),
        rain_mm    = ("rain_mm",    "sum"),
    ).reset_index()
    daily["date_label"] = pd.to_datetime(daily["date"]).dt.strftime("%b %d")

    # Condition frequency
    cond_counts = df["description"].value_counts()

    return df, daily, cond_counts


# ══════════════════════════════════════════════════════════
# 3.  DASHBOARD VISUALISATION
# ══════════════════════════════════════════════════════════

# ── Palette ───────────────────────────────────────────────
BG        = "#0d1117"
CARD      = "#161b22"
ACCENT1   = "#58a6ff"    # blue  – temperature
ACCENT2   = "#f97316"    # orange – feels-like / high
ACCENT3   = "#22d3ee"    # cyan  – humidity
ACCENT4   = "#a78bfa"    # violet – wind
ACCENT5   = "#34d399"    # green – rain
GRID_CLR  = "#21262d"
TEXT_CLR  = "#e6edf3"
SUBTEXT   = "#8b949e"

PALETTE   = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
             "#fb7185", "#fbbf24", "#86efac"]


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT_CLR, fontsize=10, fontweight="bold",
                     pad=8, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


def build_dashboard(current, df, daily, cond_counts):
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.subplots_adjust(hspace=0.42, wspace=0.32,
                        left=0.05, right=0.97, top=0.90, bottom=0.06)

    # ── Main title ────────────────────────────────────────
    fig.text(0.5, 0.95,
             f"🌤  Weather Dashboard — {current['city']}, India",
             ha="center", va="center", color=TEXT_CLR,
             fontsize=18, fontweight="bold")
    fig.text(0.5, 0.925,
             f"OpenWeatherMap API Integration  •  Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}",
             ha="center", va="center", color=SUBTEXT, fontsize=9)

    gs = gridspec.GridSpec(3, 4, figure=fig)

    # ── Panel 0: Current Conditions KPI cards ─────────────
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(BG)
    for s in ax_kpi.spines.values(): s.set_visible(False)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])

    kpis = [
        ("🌡  Temperature",  f"{current['temp']:.1f}°C",   ACCENT1),
        ("🤔  Feels Like",   f"{current['feels_like']:.1f}°C", ACCENT2),
        ("💧  Humidity",     f"{current['humidity']:.0f}%",  ACCENT3),
        ("💨  Wind Speed",   f"{current['wind_speed']:.1f} m/s", ACCENT4),
        ("🌤  Condition",    current['description'],        ACCENT5),
    ]
    box_w, box_h, gap = 0.16, 0.72, 0.025
    x_start = 0.025
    for i, (label, val, color) in enumerate(kpis):
        x = x_start + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch((x, 0.10), box_w, box_h,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor=color, facecolor=CARD,
            transform=ax_kpi.transAxes, zorder=2)
        ax_kpi.add_patch(rect)
        ax_kpi.text(x + box_w/2, 0.74, label, ha="center", va="center",
                    transform=ax_kpi.transAxes, color=SUBTEXT, fontsize=8)
        ax_kpi.text(x + box_w/2, 0.38, val, ha="center", va="center",
                    transform=ax_kpi.transAxes, color=color,
                    fontsize=14, fontweight="bold")

    # ── Panel 1: Temperature over time ────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.plot(df["datetime"], df["temp"],
             color=ACCENT1, lw=1.8, label="Temp (°C)", zorder=3)
    ax1.plot(df["datetime"], df["feels_like"],
             color=ACCENT2, lw=1.2, linestyle="--",
             label="Feels Like (°C)", alpha=0.8, zorder=3)
    ax1.fill_between(df["datetime"], df["temp"], df["feels_like"],
                     alpha=0.12, color=ACCENT1)
    ax1.legend(facecolor=CARD, labelcolor=TEXT_CLR,
               fontsize=8, framealpha=0.8)
    ax1.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%b %d\n%H:%M"))
    style_ax(ax1, "📈  Temperature Forecast (5-day)",
             ylabel="°C")

    # ── Panel 2: Humidity heatmap by day × hour ───────────
    ax2 = fig.add_subplot(gs[1, 2:])
    pivot = df.pivot_table(values="humidity",
                           index="hour", columns="day_label",
                           aggfunc="mean")
    sns.heatmap(pivot, ax=ax2, cmap="YlGnBu",
                linewidths=0.3, linecolor=BG,
                annot=True, fmt=".0f", annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    ax2.set_facecolor(CARD)
    ax2.set_title("💧  Humidity (%) — Day × Hour",
                  color=TEXT_CLR, fontsize=10, fontweight="bold",
                  pad=8, loc="left")
    ax2.tick_params(colors=SUBTEXT, labelsize=7)
    ax2.set_xlabel("Date", color=SUBTEXT, fontsize=8)
    ax2.set_ylabel("Hour of Day", color=SUBTEXT, fontsize=8)

    # ── Panel 3: Daily temp range (bar + scatter) ─────────
    ax3 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(daily))
    bars = ax3.bar(x, daily["temp_max"] - daily["temp_min"],
                   bottom=daily["temp_min"],
                   color=ACCENT1, alpha=0.55, width=0.5,
                   label="Temp Range")
    ax3.scatter(x, daily["temp_max"], color=ACCENT2,
                zorder=5, s=55, label="Max")
    ax3.scatter(x, daily["temp_min"], color=ACCENT3,
                zorder=5, s=55, label="Min")
    ax3.set_xticks(x)
    ax3.set_xticklabels(daily["date_label"], rotation=30, ha="right")
    ax3.legend(facecolor=CARD, labelcolor=TEXT_CLR,
               fontsize=7, framealpha=0.8)
    style_ax(ax3, "🌡  Daily Temp Range", ylabel="°C")

    # ── Panel 4: Wind speed line ───────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.fill_between(df["datetime"], df["wind_speed"],
                     color=ACCENT4, alpha=0.3)
    ax4.plot(df["datetime"], df["wind_speed"],
             color=ACCENT4, lw=1.5)
    ax4.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%b %d"))
    plt.setp(ax4.xaxis.get_majorticklabels(),
             rotation=30, ha="right")
    style_ax(ax4, "💨  Wind Speed", ylabel="m/s")

    # ── Panel 5: Rainfall bar chart ───────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.bar(daily["date_label"], daily["rain_mm"],
            color=ACCENT5, alpha=0.8, width=0.5)
    ax5.set_xticklabels(daily["date_label"], rotation=30, ha="right")
    style_ax(ax5, "🌧  Daily Rainfall", ylabel="mm")

    # ── Panel 6: Weather condition distribution ───────────
    ax6 = fig.add_subplot(gs[2, 3])
    colors_pie = PALETTE[:len(cond_counts)]
    wedges, texts, autotexts = ax6.pie(
        cond_counts.values,
        labels=None,
        autopct="%1.0f%%",
        colors=colors_pie,
        startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(linewidth=1.5, edgecolor=BG)
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(7.5)
        at.set_fontweight("bold")
    ax6.set_facecolor(CARD)
    ax6.set_title("🌈  Condition Split",
                  color=TEXT_CLR, fontsize=10, fontweight="bold",
                  pad=8, loc="left")
    ax6.legend(wedges, cond_counts.index, loc="lower center",
               bbox_to_anchor=(0.5, -0.18),
               facecolor=CARD, labelcolor=TEXT_CLR,
               fontsize=6.5, ncol=2, framealpha=0.8)

    # ── Footer ────────────────────────────────────────────
    fig.text(0.5, 0.01,
             "Data Source: OpenWeatherMap API  •  CODTECH Internship Project  •  "
             "Tools: Python, Requests, Pandas, Matplotlib, Seaborn",
             ha="center", color=SUBTEXT, fontsize=7.5)

    return fig


# ══════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1 — fetch data
    current, df_raw = get_weather_data()

    # Step 2 — process
    df, daily, cond_counts = process(df_raw)

    # Step 3 — print summary to console
    print("\n" + "═" * 55)
    print(f"  CURRENT WEATHER — {current['city']}")
    print("═" * 55)
    for k, v in current.items():
        if k != "city":
            label = k.replace("_", " ").title()
            print(f"  {label:<15}: {v}")
    print("═" * 55)
    print(f"\n  5-Day Forecast: {len(df)} records loaded")
    print(f"  Temperature range: {df['temp'].min():.1f}°C – {df['temp'].max():.1f}°C")
    print(f"  Average humidity : {df['humidity'].mean():.1f}%")
    print(f"  Peak wind speed  : {df['wind_speed'].max():.1f} m/s")
    print(f"  Total rainfall   : {df['rain_mm'].sum():.1f} mm")
    print("═" * 55 + "\n")

    # Step 4 — build & save dashboard
    print("🎨  Building dashboard …")
    fig = build_dashboard(current, df, daily, cond_counts)
    out = "weather_dashboard.png"
    fig.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"✅  Dashboard saved → {out}")
