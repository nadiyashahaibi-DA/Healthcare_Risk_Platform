"""
Shared color palette and small styling helpers for the Healthcare Risk
Intelligence Portal.

Keeping every color in ONE place (instead of picking colors separately on
each page) is what makes the app look like one coherent product instead of
five different charts glued together. The same hex codes are documented in
Streamlit_app/README.md so they can be reused in Tableau and the summary
slide deck, for a consistent look across all three deliverables.

Color roles used in this app:
- BRAND_BLUE is the single "identity" color: headers, buttons, the default
  chart color whenever a chart is just showing one series.
- The RISK_* colors are a fixed "status" palette reserved ONLY for
  low/medium/high readmission risk. They are never reused as a general
  chart color, so a reader always knows "colored red/amber/green" means
  risk level and nothing else.
- SEQUENTIAL_BLUES is a light-to-dark ramp of the brand blue, used for
  charts that rank a value from low to high (e.g. "readmission rate by
  age group") - one hue, darker = higher, instead of a different color per
  bar (which would wrongly suggest the bars are unrelated categories).
"""

# ---- Brand / identity ---------------------------------------------------
BRAND_BLUE = "#2a78d6"
BRAND_BLUE_DARK = "#184f95"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
SURFACE_ALT = "#eef2f6"
GRIDLINE = "#e1e0d9"

# ---- Status palette (risk tiers) — fixed, never reused for anything else
RISK_LOW = "#0ca30c"     # good / low risk
RISK_MEDIUM = "#fab219"  # warning / medium risk
RISK_HIGH = "#d03b3b"    # critical / high risk
RISK_NEUTRAL = "#898781"  # "not readmitted" / baseline comparison group

# ---- Sequential ramp (magnitude: rates, counts, rankings) ---------------
SEQUENTIAL_BLUES = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

# ---- Diverging pair (polarity: "raises risk" vs. "lowers risk") ---------
DIVERGING_RAISES_RISK = "#d03b3b"   # red
DIVERGING_LOWERS_RISK = "#2a78d6"   # blue
DIVERGING_NEUTRAL = "#f0efec"


def risk_tier(probability: float) -> str:
    """Bucket a predicted readmission probability into Low / Medium / High.

    Thresholds are a simple, explainable rule (not learned from data) -
    consistent with the rest of this project's preference for transparent,
    easy-to-explain logic over black-box scoring.
    """
    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Medium"
    return "High"


def risk_color(tier: str) -> str:
    return {"Low": RISK_LOW, "Medium": RISK_MEDIUM, "High": RISK_HIGH}[tier]


def risk_icon(tier: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[tier]


def inject_global_css() -> str:
    """Returns a <style> block applied on every page for a consistent look:
    a clean sans-serif font, a soft page background, and card-style metrics.
    Call with st.markdown(inject_global_css(), unsafe_allow_html=True).
    """
    return f"""
    <style>
        html, body, [class*="css"] {{
            font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
        }}
        .block-container {{
            padding-top: 2rem;
        }}
        h1, h2, h3 {{
            color: {INK_PRIMARY};
        }}
        [data-testid="stMetric"] {{
            background-color: {SURFACE_ALT};
            border: 1px solid {GRIDLINE};
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {INK_SECONDARY};
        }}
        .risk-badge {{
            border-radius: 12px;
            padding: 22px 24px;
            text-align: center;
            color: white;
            font-size: 1.1rem;
        }}
        .risk-badge .tier {{
            font-size: 2.0rem;
            font-weight: 700;
            margin: 4px 0;
        }}
    </style>
    """


def risk_badge_html(tier: str, probability: float) -> str:
    color = risk_color(tier)
    icon = risk_icon(tier)
    return f"""
    <div class="risk-badge" style="background-color:{color};">
        <div>{icon} Predicted 30-day readmission risk</div>
        <div class="tier">{tier.upper()}</div>
        <div>{probability:.0%} estimated probability</div>
    </div>
    """
