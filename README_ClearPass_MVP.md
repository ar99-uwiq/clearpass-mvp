
# ClearPass — AI Financial Health MVP

This MVP ingests simple financial line items, computes key ratios, compares them to industry benchmarks, generates an AI-style summary, and exports a multi-page PDF report.

## Files
- `clearpass_app.py` — Streamlit app
- `sample_financials.csv` — Sample data to test
- `industry_benchmarks.csv` — Simple benchmark medians by industry

## How to Run
1. **Install dependencies**
   ```bash
   pip install streamlit pandas matplotlib
   ```
   *(Optional)* If you want real AI summaries, set `OPENAI_API_KEY` in your environment and add your preferred LLM client.

2. **Start the app**
   ```bash
   streamlit run clearpass_app.py
   ```

3. **Use the app**
   - Upload `sample_financials.csv` (provided).
   - Choose an industry in the sidebar.
   - View KPIs, benchmark chart, and AI-like summary.
   - Click **Generate PDF Report** to export a multi-page PDF.

## Data Format
Expect a CSV/XLSX with:
- Column **Account** (e.g., Current Assets, Current Liabilities, Total Liabilities, Equity, Total Assets, Revenue, Net Income, Inventory, Cash, Accounts Receivable, Interest Expense)
- Column **Value** (numeric)

## Notes
- The **QuickBooks integration section** is a **stub** showing how you would connect. It’s disabled in the MVP.
- The AI summary currently uses a **rule-based** narrative unless you set `OPENAI_API_KEY` in your environment.
- Charts are built with Matplotlib only (no seaborn), one chart per page, and no custom colors.

## Next Steps
- Add authentication & Stripe billing
- Expand ratios (coverage, turnover, cash conversion cycle)
- Add XLSX export & white-label templates
