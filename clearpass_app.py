
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# ---------- APP CONFIG ----------
st.set_page_config(page_title="ClearPass — AI Financial Health MVP", layout="wide")
st.title("💡 ClearPass — AI Financial Health MVP")
st.caption("Upload basic financials, compare to industry benchmarks, get an AI-written summary, and export a PDF.")

# ---------- SIDEBAR: Industry & Settings ----------
st.sidebar.header("Settings")
industry_df = pd.read_csv("industry_benchmarks.csv")
industry_options = industry_df["industry_name"].tolist()
industry = st.sidebar.selectbox("Industry (for benchmarks)", industry_options, index=0)
selected_bench = industry_df[industry_df["industry_name"] == industry].iloc[0].to_dict()

st.sidebar.markdown("""
**Expected columns (CSV/XLSX):**
- `Account` (e.g., 'Current Assets', 'Revenue', 'Inventory', 'Total Liabilities', 'Equity', 'Net Income', 'Total Assets', 'Current Liabilities')
- `Value` (numeric)
Optionally include: 'Cash', 'Accounts Receivable', 'Inventory', 'Interest Expense', etc.
""")

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader("Upload Financials (CSV or Excel)", type=["csv", "xlsx"])

def _read_df(uploaded):
    if uploaded is None:
        return None
    try:
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded)
        return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

df = _read_df(uploaded)

# ---------- HELPER: SAFE GET SUM BY ACCOUNT KEYWORD ----------
def get_sum(df, keyword_list):
    if df is None:
        return 0.0
    mask = pd.Series(False, index=df.index)
    for kw in keyword_list:
        mask = mask | df["Account"].astype(str).str.contains(kw, case=False, regex=True)
    return pd.to_numeric(df.loc[mask, "Value"], errors="coerce").fillna(0).sum()

# ---------- RATIO CALCULATIONS ----------
def compute_ratios(df):
    current_assets = get_sum(df, ["^current assets$"])
    current_liabilities = get_sum(df, ["^current liabilities$"])
    total_liabilities = get_sum(df, ["^total liabilities$"])
    equity = get_sum(df, ["^equity$", "shareholders' equity", "total equity"])
    total_assets = get_sum(df, ["^total assets$"])
    revenue = get_sum(df, ["^revenue$", "^sales$"])
    net_income = get_sum(df, ["^net income$", "net profit"])
    inventory = get_sum(df, ["^inventory$"])
    cash = get_sum(df, ["^cash$"])
    ar = get_sum(df, ["accounts receivable"])
    interest_expense = get_sum(df, ["interest expense"])

    # Derive quick assets (if explicit cash+AR available, prefer that)
    quick_assets = cash + ar if (cash > 0 or ar > 0) else max(current_assets - inventory, 0)

    ratios = {}
    ratios["Current Ratio"] = round(current_assets / current_liabilities, 2) if current_liabilities else None
    ratios["Quick Ratio"] = round(quick_assets / current_liabilities, 2) if current_liabilities else None
    ratios["Debt-to-Equity"] = round(total_liabilities / equity, 2) if equity else None
    ratios["Profit Margin (%)"] = round((net_income / revenue) * 100, 2) if revenue else None
    ratios["Return on Assets (%)"] = round((net_income / total_assets) * 100, 2) if total_assets else None
    # Optional coverage ratio if inputs exist
    ratios["Interest Coverage (x)"] = round((net_income + interest_expense) / interest_expense, 2) if interest_expense else None

    basics = {
        "Current Assets": current_assets,
        "Current Liabilities": current_liabilities,
        "Total Liabilities": total_liabilities,
        "Equity": equity,
        "Total Assets": total_assets,
        "Revenue": revenue,
        "Net Income": net_income,
        "Inventory": inventory,
        "Cash": cash,
        "Accounts Receivable": ar
    }
    return ratios, basics

def benchmark_compare(ratios, bench_row):
    bench = {
        "Current Ratio": bench_row.get("current_ratio_median", np.nan),
        "Quick Ratio": bench_row.get("quick_ratio_median", np.nan),
        "Debt-to-Equity": bench_row.get("d_to_e_median", np.nan),
        "Profit Margin (%)": bench_row.get("profit_margin_median", np.nan),
        "Return on Assets (%)": bench_row.get("roa_median", np.nan)
    }
    return bench

def ai_summary(ratios, basics, industry_name):
    # AI stub: if OPENAI_API_KEY set, use it outside this MVP; otherwise rule-based narrative.
    key = os.environ.get("OPENAI_API_KEY", "")
    base_summary = []
    # Rule-based text
    cr = ratios.get("Current Ratio")
    qr = ratios.get("Quick Ratio")
    de = ratios.get("Debt-to-Equity")
    pm = ratios.get("Profit Margin (%)")
    roa = ratios.get("Return on Assets (%)")

    def level(val, good_thr, ok_thr, inv=False):
        if val is None:
            return "n/a"
        if not inv:
            if val >= good_thr: return "strong"
            if val >= ok_thr: return "acceptable"
            return "weak"
        else:
            # inverse (lower is better)
            if val <= good_thr: return "strong"
            if val <= ok_thr: return "acceptable"
            return "elevated"

    base_summary.append(f"**Liquidity:** Current Ratio {cr} and Quick Ratio {qr}. Liquidity appears {level(cr if cr is not None else 0, 1.8, 1.2)} relative to common thresholds (≥1.8 strong, 1.2–1.8 acceptable). Working capital position supports near-term obligations.")
    base_summary.append(f"**Leverage:** Debt-to-Equity {de}. Leverage is {level(de if de is not None else 0, 0.8, 1.5, inv=True)} (lower is better), suggesting {'conservative' if (de is not None and de<1) else 'moderate' if (de is not None and de<2) else 'higher'} reliance on debt.")
    base_summary.append(f"**Profitability:** Profit Margin {pm}% and ROA {roa}%. Profitability is {level(pm if pm is not None else 0, 12, 6)} with returns consistent with {industry_name} peers.")
    base_summary.append("**Overall View:** Balanced financial health with attention to maintaining liquidity and disciplined leverage. Monitor cash conversion and debt service under stress scenarios.")
    return "\n\n".join(base_summary)

def plot_ratio_vs_benchmark(ratios, bench, title):
    metrics = [m for m in ["Current Ratio", "Quick Ratio", "Debt-to-Equity", "Profit Margin (%)", "Return on Assets (%)"] if (ratios.get(m) is not None) or (not np.isnan(bench.get(m, np.nan)))]
    company_vals = [ratios.get(m, np.nan) for m in metrics]
    bench_vals = [bench.get(m, np.nan) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, company_vals, width, label="Company")
    ax.bar(x + width/2, bench_vals, width, label="Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def make_pdf(company_name, year, ratios, basics, bench, industry):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Cover page
        fig1, ax1 = plt.subplots(figsize=(8.27, 11.69))
        ax1.axis("off")
        ax1.text(0.5, 0.8, "ClearPass Financial Health Report", ha="center", va="center", fontsize=20)
        ax1.text(0.5, 0.74, f"Company: {company_name}", ha="center", va="center", fontsize=12)
        ax1.text(0.5, 0.7, f"Fiscal Year: {year}", ha="center", va="center", fontsize=12)
        ax1.text(0.5, 0.66, f"Industry: {industry}", ha="center", va="center", fontsize=12)
        ax1.text(0.5, 0.58, "Generated by ClearPass MVP", ha="center", va="center", fontsize=10)
        pdf.savefig(fig1); plt.close(fig1)

        # Ratios table page
        fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
        ax2.axis("off")
        y = 0.95
        ax2.text(0.1, y, "Key Ratios", fontsize=14)
        y -= 0.04
        for k, v in ratios.items():
            ax2.text(0.1, y, f"{k}", fontsize=10)
            ax2.text(0.6, y, f"{'' if v is None else v}", fontsize=10)
            y -= 0.03
        y -= 0.02
        ax2.text(0.1, y, "Basics", fontsize=14); y -= 0.04
        for k, v in basics.items():
            ax2.text(0.1, y, f"{k}", fontsize=10)
            ax2.text(0.6, y, f"{v:,.2f}", fontsize=10)
            y -= 0.03
        pdf.savefig(fig2); plt.close(fig2)

        # Benchmark chart
        fig3 = plot_ratio_vs_benchmark(ratios, bench, "Company vs Industry Benchmark")
        pdf.savefig(fig3); plt.close(fig3)

        # AI Summary page
        summary = ai_summary(ratios, basics, industry)
        fig4, ax4 = plt.subplots(figsize=(8.27, 11.69))
        ax4.axis("off")
        ax4.text(0.1, 0.95, "AI Financial Health Summary", fontsize=14)
        import textwrap
        wrapped = textwrap.fill(summary.replace("**", ""), width=95)
        ax4.text(0.1, 0.9, wrapped, fontsize=10, va="top")
        pdf.savefig(fig4); plt.close(fig4)
    buffer.seek(0)
    return buffer

# ---------- MAIN UI ----------
if df is not None:
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Basic meta
    cols = st.columns(3)
    company_name = cols[0].text_input("Company Name", "DemoCo Ltd.")
    fiscal_year = cols[1].text_input("Fiscal Year", "2024")
    industry_name = selected_bench["industry_name"]

    ratios, basics = compute_ratios(df)
    bench = benchmark_compare(ratios, selected_bench)

    # KPI Cards
    st.subheader("📈 Key Ratios")
    kpi_cols = st.columns(5)
    metrics = ["Current Ratio", "Quick Ratio", "Debt-to-Equity", "Profit Margin (%)", "Return on Assets (%)"]
    for i, m in enumerate(metrics):
        with kpi_cols[i % 5]:
            st.metric(m, value="n/a" if ratios.get(m) is None else ratios[m])

    # Chart
    st.subheader("📊 Benchmark Comparison")
    fig = plot_ratio_vs_benchmark(ratios, bench, f"{company_name} vs {industry_name}")
    st.pyplot(fig)

    # AI Summary
    st.subheader("🧠 AI Financial Health Summary")
    summary_text = ai_summary(ratios, basics, industry_name)
    st.markdown(summary_text)

    # PDF Export
    st.subheader("📄 Export")
    if st.button("Generate PDF Report"):
        pdf_buffer = make_pdf(company_name, fiscal_year, ratios, basics, bench, industry_name)
        st.download_button("Download PDF", data=pdf_buffer, file_name=f"{company_name}_Financial_Health_Report.pdf")

else:
    st.info("Upload your financials to begin. Need a sample? Download the sample CSV from below.")

st.markdown("---")
st.header("🧪 QuickBooks Integration (Stub)")
with st.expander("Show QuickBooks API integration outline"):
    st.markdown("""
**This is a stub for future integration:**
1. OAuth2 with Intuit (QuickBooks Online) - obtain access/refresh tokens.
2. Use QBO endpoints: `/v3/company/<realmId>/query` to pull Profit & Loss and Balance Sheet.
3. Transform into the ClearPass schema (Account, Value).
4. Auto-refresh data monthly and re-run analysis.

**Example (pseudo-code):**
```python
from quickbooks import QuickBooks
from quickbooks.objects.report import Report

qbo = QuickBooks(
    sandbox=True,
    consumer_key="...",
    consumer_secret="...",
    access_token="...",
    access_token_secret="...",
    company_id="REALM_ID"
)

pl = Report.get("ProfitAndLoss", qb=qbo)
bs = Report.get("BalanceSheet", qb=qbo)
# Map to rows: [{'Account': 'Revenue', 'Value': ...}, ...]
```
""")
