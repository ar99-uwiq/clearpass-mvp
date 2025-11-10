
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================
# App Config
# ==========================
st.set_page_config(page_title="ClearPass — Financial Health", layout="wide")
st.title("💡 ClearPass — Financial Health")
st.caption("Upload financials, benchmark performance, generate a branded PDF report.")

# ==========================
# Helpers: default data (fallbacks so files are never 'not found')
# ==========================
DEFAULT_BENCHMARKS = """naics,industry_name,current_ratio_median,quick_ratio_median,d_to_e_median,profit_margin_median,roa_median
311,Food Manufacturing,1.5,1.2,1.2,8.0,6.0
541,Professional Services,1.8,1.6,0.8,12.0,10.0
423,Wholesale Trade,1.6,1.3,1.0,6.0,5.0
"""
DEFAULT_SAMPLE = """Account,Value
Current Assets,150000
Current Liabilities,81000
Total Liabilities,220000
Equity,320000
Total Assets,540000
Revenue,1200000
COGS,720000
Operating Expenses,300000
Net Income,150000
Inventory,30000
Cash,60000
Accounts Receivable,40000
Interest Expense,20000
"""

def ensure_file(path, content):
    """Create file if it doesn't exist so users don't hit 'file not found'."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)

APP_DIR = os.getcwd()
BENCH_PATH = os.path.join(APP_DIR, "industry_benchmarks.csv")
SAMPLE_PATH = os.path.join(APP_DIR, "sample_financials.csv")
ensure_file(BENCH_PATH, DEFAULT_BENCHMARKS)
ensure_file(SAMPLE_PATH, DEFAULT_SAMPLE)

# ==========================
# Sidebar: Data + Branding
# ==========================
st.sidebar.header("⚙️ Settings")

# Industry Benchmarks (auto-loaded or fallback)
try:
    industry_df = pd.read_csv(BENCH_PATH)
except Exception:
    industry_df = pd.read_csv(io.StringIO(DEFAULT_BENCHMARKS))

industry_options = industry_df["industry_name"].tolist()
industry = st.sidebar.selectbox("Industry (for benchmarks)", industry_options, index=0)

brand_name = st.sidebar.text_input("Brand name", "ClearPass")
primary_color = st.sidebar.color_picker("Primary color", "#1F4B99")
accent_color = st.sidebar.color_picker("Accent color", "#00B894")
uploaded_logo = st.sidebar.file_uploader("Upload logo (PNG/JPG, optional)", type=["png", "jpg", "jpeg"])

st.sidebar.markdown("### Data Schema")
st.sidebar.markdown("""
**Expected columns (CSV/XLSX):**
- **Account** (e.g., Current Assets, Revenue, Total Liabilities, Equity, Net Income, Total Assets, Current Liabilities, Inventory, Cash, Accounts Receivable, Interest Expense)
- **Value** (numeric)
""")
st.sidebar.markdown("Need test data? Download the sample below.")
st.sidebar.download_button("⬇️ Download sample_financials.csv", data=DEFAULT_SAMPLE, file_name="sample_financials.csv")
st.sidebar.download_button("⬇️ Download industry_benchmarks.csv", data=DEFAULT_BENCHMARKS, file_name="industry_benchmarks.csv")

# ==========================
# Upload Section
# ==========================
uploaded = st.file_uploader("Upload Financials (CSV or Excel)", type=["csv", "xlsx"])

def read_df(uploaded):
    if uploaded is None:
        return None
    try:
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded)
        return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

df = read_df(uploaded)

# ==========================
# Ratio Calculations
# ==========================
def get_sum(df, keyword_list):
    if df is None or "Account" not in df.columns or "Value" not in df.columns:
        return 0.0
    mask = pd.Series(False, index=df.index)
    for kw in keyword_list:
        mask = mask | df["Account"].astype(str).str.contains(kw, case=False, regex=True)
    return pd.to_numeric(df.loc[mask, "Value"], errors="coerce").fillna(0).sum()

def compute_ratios(df):
    current_assets = get_sum(df, [r"^current assets$"])
    current_liabilities = get_sum(df, [r"^current liabilities$"])
    total_liabilities = get_sum(df, [r"^total liabilities$"])
    equity = get_sum(df, [r"^equity$", r"shareholders' equity", r"total equity"])
    total_assets = get_sum(df, [r"^total assets$"])
    revenue = get_sum(df, [r"^revenue$", r"^sales$"])
    net_income = get_sum(df, [r"^net income$", r"net profit"])
    inventory = get_sum(df, [r"^inventory$"])
    cash = get_sum(df, [r"^cash$"])
    ar = get_sum(df, [r"accounts receivable"])
    interest_expense = get_sum(df, [r"interest expense"])

    quick_assets = cash + ar if (cash > 0 or ar > 0) else max(current_assets - inventory, 0)

    ratios = {
        "Current Ratio": round(current_assets / current_liabilities, 2) if current_liabilities else None,
        "Quick Ratio": round(quick_assets / current_liabilities, 2) if current_liabilities else None,
        "Debt-to-Equity": round(total_liabilities / equity, 2) if equity else None,
        "Profit Margin (%)": round((net_income / revenue) * 100, 2) if revenue else None,
        "Return on Assets (%)": round((net_income / total_assets) * 100, 2) if total_assets else None,
        "Interest Coverage (x)": round((net_income + interest_expense) / interest_expense, 2) if interest_expense else None,
    }
    basics = {
        "Revenue": revenue,
        "Net Income": net_income,
        "Total Assets": total_assets,
        "Total Liabilities": total_liabilities,
        "Equity": equity,
        "Current Assets": current_assets,
        "Current Liabilities": current_liabilities,
        "Cash": cash,
        "Accounts Receivable": ar,
        "Inventory": inventory,
    }
    return ratios, basics

def get_bench_row(ind_name):
    row = industry_df[industry_df["industry_name"] == ind_name].iloc[0].to_dict()
    bench = {
        "Current Ratio": row.get("current_ratio_median", np.nan),
        "Quick Ratio": row.get("quick_ratio_median", np.nan),
        "Debt-to-Equity": row.get("d_to_e_median", np.nan),
        "Profit Margin (%)": row.get("profit_margin_median", np.nan),
        "Return on Assets (%)": row.get("roa_median", np.nan),
    }
    return row, bench

def ai_like_summary(ratios, industry_name):
    cr = ratios.get("Current Ratio")
    qr = ratios.get("Quick Ratio")
    de = ratios.get("Debt-to-Equity")
    pm = ratios.get("Profit Margin (%)")
    roa = ratios.get("Return on Assets (%)")

    def level(val, good, ok, inv=False):
        if val is None: return "n/a"
        if not inv:
            if val >= good: return "strong"
            if val >= ok: return "acceptable"
            return "weak"
        else:
            if val <= good: return "strong"
            if val <= ok: return "acceptable"
            return "elevated"

    parts = []
    parts.append(f"**Liquidity** — Current Ratio {cr}, Quick Ratio {qr}. Liquidity appears {level(cr or 0, 1.8, 1.2)} relative to typical thresholds (≥1.8 strong; 1.2–1.8 acceptable).")
    parts.append(f"**Leverage** — Debt-to-Equity {de}. Leverage is {level(de or 0, 0.8, 1.5, inv=True)} (lower is better) indicating {'conservative' if (de is not None and de<1) else 'moderate' if (de is not None and de<2) else 'higher'} reliance on debt.")
    parts.append(f"**Profitability** — Profit Margin {pm}% and ROA {roa}%. Profitability is {level(pm or 0, 12, 6)} versus peers in {industry_name}.")
    parts.append("**Overall** — Balanced profile with attention to working capital discipline and debt service resilience. Monitor cash conversion and maintain adequate interest coverage.")
    return "\n\n".join(parts)

def plot_benchmark_bars(ratios, bench, title):
    metrics = [m for m in ["Current Ratio","Quick Ratio","Debt-to-Equity","Profit Margin (%)","Return on Assets (%)"]
               if (ratios.get(m) is not None) or (not np.isnan(bench.get(m, np.nan)))]
    company_vals = [ratios.get(m, np.nan) for m in metrics]
    bench_vals = [bench.get(m, np.nan) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, company_vals, width, label="Company")
    ax.bar(x + width/2, bench_vals, width, label="Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

# ==========================
# Branded PDF
# ==========================
def branded_pdf_to_buffer(company_name, fiscal_year, industry_name, ratios, basics, bench,
                          brand="ClearPass", primary="#1F4B99", accent="#00B894", logo_img=None):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color=primary, transform=ax.transAxes))
        if logo_img is not None:
            ax_logo = fig.add_axes([0.08, 0.90, 0.18, 0.08]); ax_logo.axis("off"); ax_logo.imshow(logo_img); ax_logo.set_aspect('auto')
        ax.text(0.28, 0.965, f"{brand} — Financial Health Report", fontsize=18, weight="bold", color="white", transform=ax.transAxes, va="center")
        ax.text(0.08, 0.84, company_name, fontsize=24, weight='bold', color="#222")
        ax.text(0.08, 0.80, f"Fiscal Year: {fiscal_year}", fontsize=12, color="#666")
        ax.text(0.08, 0.77, f"Industry: {industry_name}", fontsize=12, color="#666")
        ax.add_patch(plt.Rectangle((0.08, 0.73), 0.84, 0.003, color=accent))
        ax.text(0.08, 0.69, "Overview", fontsize=14, weight="bold", color="#222")
        overview = "This report summarizes liquidity, leverage, profitability, and overall financial health, and compares performance to industry medians."
        import textwrap
        ax.text(0.08, 0.665, textwrap.fill(overview, 95), fontsize=10, color="#222")
        pdf.savefig(fig); plt.close(fig)

        # Ratios page
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color=primary, transform=ax.transAxes))
        ax.text(0.08, 0.965, "Key Ratios & Fundamentals", fontsize=16, weight="bold", color="white", transform=ax.transAxes, va="center")
        y = 0.86
        ax.text(0.08, y, "Key Ratios", fontsize=14, weight="bold", color="#222"); y -= 0.035
        for k in ["Current Ratio","Quick Ratio","Debt-to-Equity","Profit Margin (%)","Return on Assets (%)","Interest Coverage (x)"]:
            v = ratios.get(k, None)
            ax.text(0.08, y, f"{k}", fontsize=11, color="#222")
            ax.text(0.65, y, f"{'—' if v is None else v}", fontsize=11, color="#222")
            y -= 0.03
        y -= 0.02
        ax.add_patch(plt.Rectangle((0.08, y), 0.84, 0.002, color="#999")); y -= 0.04
        ax.text(0.08, y, "Fundamentals", fontsize=14, weight="bold", color="#222"); y -= 0.035
        for k in ["Revenue","Net Income","Total Assets","Total Liabilities","Equity","Current Assets","Current Liabilities","Cash","Accounts Receivable","Inventory"]:
            v = basics.get(k, 0.0)
            ax.text(0.08, y, f"{k}", fontsize=10, color="#222")
            ax.text(0.65, y, f"{v:,.2f}", fontsize=10, color="#222")
            y -= 0.027
        pdf.savefig(fig); plt.close(fig)

        # Benchmark chart
        metrics = [m for m in ["Current Ratio","Quick Ratio","Debt-to-Equity","Profit Margin (%)","Return on Assets (%)"]
                   if (ratios.get(m) is not None) or (not np.isnan(bench.get(m, np.nan)))]
        company_vals = [ratios.get(m, np.nan) for m in metrics]
        bench_vals = [bench.get(m, np.nan) for m in metrics]
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color=primary, transform=ax.transAxes))
        ax.text(0.08, 0.965, "Company vs Industry Benchmark", fontsize=16, weight="bold", color="white", transform=ax.transAxes, va="center")
        ax2 = fig.add_axes([0.08, 0.12, 0.84, 0.7])
        x = np.arange(len(metrics)); width = 0.35
        ax2.bar(x - width/2, company_vals, width, label="Company")
        ax2.bar(x + width/2, bench_vals, width, label="Benchmark")
        ax2.set_xticks(x); ax2.set_xticklabels(metrics, rotation=15, ha="right")
        ax2.legend()
        pdf.savefig(fig); plt.close(fig)

        # AI-like summary
        cr = ratios.get("Current Ratio"); qr = ratios.get("Quick Ratio")
        de = ratios.get("Debt-to-Equity"); pm = ratios.get("Profit Margin (%)"); roa = ratios.get("Return on Assets (%)")
        def level(val, good, ok, inv=False):
            if val is None: return "n/a"
            if not inv:
                if val >= good: return "strong"
                if val >= ok: return "acceptable"
                return "weak"
            else:
                if val <= good: return "strong"
                if val <= ok: return "acceptable"
                return "elevated"
        narrative = []
        narrative.append(f"Liquidity: Current Ratio {cr} and Quick Ratio {qr}. Liquidity appears {level(cr or 0, 1.8, 1.2)} relative to common thresholds.")
        narrative.append(f"Leverage: Debt-to-Equity {de}. Leverage is {level(de or 0, 0.8, 1.5, inv=True)}; lower is generally better.")
        narrative.append(f"Profitability: Profit Margin {pm}% and ROA {roa}%. Profitability is {level(pm or 0, 12, 6)} versus peers in {industry_name}.")
        narrative.append("Overall: Balanced financial health with attention to working capital and interest coverage under stress.")
        summary = "\n\n".join(narrative)
        import textwrap
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color=primary, transform=ax.transAxes))
        ax.text(0.08, 0.965, "AI Financial Health Summary", fontsize=16, weight="bold", color="white", transform=ax.transAxes, va="center")
        ax.text(0.08, 0.90, textwrap.fill(summary, 98), fontsize=11, color="#222", va="top")
        pdf.savefig(fig); plt.close(fig)
    buf.seek(0)
    return buf

# ==========================
# Main
# ==========================
if df is not None:
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    cols = st.columns(3)
    company_name = cols[0].text_input("Company Name", "DemoCo Ltd.")
    fiscal_year = cols[1].text_input("Fiscal Year", "2024")

    ratios, basics = compute_ratios(df)
    bench_row, bench = get_bench_row(industry)

    # KPIs
    st.subheader("📈 Key Ratios")
    kpi_cols = st.columns(5)
    for i, m in enumerate(["Current Ratio","Quick Ratio","Debt-to-Equity","Profit Margin (%)","Return on Assets (%)"]):
        with kpi_cols[i]:
            st.metric(m, value="n/a" if ratios.get(m) is None else ratios[m])

    # Chart
    st.subheader("📊 Benchmark Comparison")
    fig = plot_benchmark_bars(ratios, bench, f"{company_name} vs {industry}")
    st.pyplot(fig)

    # AI-like summary
    st.subheader("🧠 AI Financial Health Summary")
    st.markdown(ai_like_summary(ratios, industry))

    # Branded PDF Export
    st.subheader("📄 Export")
    logo_img = None
    if uploaded_logo is not None:
        try:
            logo_img = Image.open(uploaded_logo).convert("RGBA")
        except Exception:
            st.warning("Could not read the uploaded logo; continuing without it.")
    if st.button("Generate Branded PDF"):
        pdf_buf = branded_pdf_to_buffer(
            company_name, fiscal_year, industry, ratios, basics, bench,
            brand=brand_name, primary=primary_color, accent=accent_color, logo_img=logo_img
        )
        st.download_button("⬇️ Download Branded PDF", data=pdf_buf, file_name=f"{company_name}_Financial_Health_Report.pdf")
else:
    st.info("Upload your CSV/XLSX to begin. Need a test file? Use the download buttons in the sidebar.")
