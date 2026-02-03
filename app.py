# app.py
# Plotly Dash "CBU Tracker" app with What-If parameters
#
# Run:
#   pip install dash pandas plotly
#   python app.py
#
# Then open: http://127.0.0.1:8050

from __future__ import annotations

import base64
import io
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
from datetime import date, datetime, timedelta

# ---- CONFIG ----
# Column names used internally
COL_PP = "PP#"
COL_FY = "Y"  # fiscal year column
COL_START = "Pay Period Start"
COL_END = "Pay Period End"
COL_WORK = "Working Hours"
COL_BILL = "Billable Hours"
COL_BILL_YTD = "Billable Hours (YTD)"


def generate_fy_pay_periods(fy: int) -> pd.DataFrame:
    """
    Generate pay period data for a fiscal year.
    FY runs from April 1 of the prior year to March 31.
    There are 26 pay periods per year (bi-weekly).
    """
    # FY26 starts Apr 1, 2025 and ends Mar 31, 2026
    fy_start = datetime(fy - 1, 4, 1)  # Apr 1 of prior calendar year
    
    pay_periods = []
    current_start = fy_start
    
    for pp in range(1, 27):  # 26 pay periods
        pp_end = current_start + timedelta(days=13)  # 14-day pay period
        
        # Standard working hours per pay period (80 hours = 2 weeks x 40 hours)
        holidays_in_pp = count_federal_holidays(current_start.date(), pp_end.date())
        working_hours = max(0.0, 80.0 - (holidays_in_pp * 8.0))
        
        pay_periods.append({
            COL_PP: pp,
            COL_START: current_start,
            COL_END: pp_end,
            COL_WORK: working_hours,
            COL_BILL: 0.0,  # User will enter this
        })
        
        current_start = pp_end + timedelta(days=1)
    
    return pd.DataFrame(pay_periods)


def count_federal_holidays(start_date: date, end_date: date) -> int:
    """Return the number of observed U.S. federal holidays in the date range (inclusive)."""
    years = {start_date.year, end_date.year}
    holiday_dates = set()
    for year in years:
        holiday_dates.update(get_us_federal_holidays(year))
    return sum(1 for holiday in holiday_dates if start_date <= holiday <= end_date)


def get_us_federal_holidays(year: int) -> set[date]:
    """Observed U.S. federal holidays for a given calendar year."""
    def nth_weekday_of_month(month: int, weekday: int, n: int) -> date:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + (n - 1) * 7)

    def last_weekday_of_month(month: int, weekday: int) -> date:
        next_month = date(year, month, 28) + timedelta(days=4)
        last_day = next_month.replace(day=1) - timedelta(days=1)
        offset = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=offset)

    def observed(dt: date) -> date:
        if dt.weekday() == 5:
            return dt - timedelta(days=1)
        if dt.weekday() == 6:
            return dt + timedelta(days=1)
        return dt

    holidays = {
        observed(date(year, 1, 1)),   # New Year's Day
        nth_weekday_of_month(1, 0, 3),  # MLK Day (3rd Monday of Jan)
        nth_weekday_of_month(2, 0, 3),  # Presidents Day (3rd Monday of Feb)
        last_weekday_of_month(5, 0),    # Memorial Day (last Monday of May)
        observed(date(year, 6, 19)),  # Juneteenth
        observed(date(year, 7, 4)),   # Independence Day
        nth_weekday_of_month(9, 0, 1),  # Labor Day (1st Monday of Sep)
        nth_weekday_of_month(10, 0, 2),  # Columbus/Indigenous Peoples Day
        observed(date(year, 11, 11)),  # Veterans Day
        nth_weekday_of_month(11, 3, 4),  # Thanksgiving (4th Thursday of Nov)
        observed(date(year, 12, 25)),  # Christmas
    }
    return holidays


def generate_sample_data() -> pd.DataFrame:
    """
    Generate sample pay period data for multiple fiscal years.
    """
    fiscal_years = [2025, 2026]  # FY25 and FY26
    all_data = []
    
    for fy in fiscal_years:
        fy_data = generate_fy_pay_periods(fy)
        fy_data["FY"] = fy
        all_data.append(fy_data)
    
    return pd.concat(all_data, ignore_index=True)


def compute_cbu(df_fy: pd.DataFrame, adj_billable_per_period: float) -> pd.DataFrame:
    """
    Compute adjusted billable, period CBU, and YTD CBU.
    Interpretation: adjustment is applied PER PAY PERIOD.
    """
    out = df_fy.copy()

    out["Billable Hours (Adj)"] = out[COL_BILL] + float(adj_billable_per_period)
    out["Billable Hours (Adj)"] = out["Billable Hours (Adj)"].clip(lower=0)

    out["CBU (Period)"] = np.where(
        out[COL_WORK] > 0,
        out["Billable Hours (Adj)"] / out[COL_WORK],
        np.nan,
    )

    out["Billable (Adj) YTD"] = out["Billable Hours (Adj)"].cumsum()
    out["Working YTD"] = out[COL_WORK].cumsum()

    out["CBU (YTD)"] = np.where(
        out["Working YTD"] > 0,
        out["Billable (Adj) YTD"] / out["Working YTD"],
        np.nan,
    )

    return out


# ---- Generate data at startup (no Excel file needed) ----
df_all = generate_sample_data()

fy_values = sorted([int(x) for x in df_all["FY"].dropna().unique().tolist()])

# ---- Dash app ----
app = Dash(__name__)
app.title = "CBU Tracker"

# Expose the Flask server for gunicorn
server = app.server

# Prepare initial billable hours data for editing
def get_initial_billable_data(fy: int) -> list:
    """Get pay periods for a fiscal year with editable billable hours."""
    dff = df_all[df_all["FY"] == fy].copy()
    dff = dff.sort_values(COL_PP).reset_index(drop=True)
    return [
        {
            "PP#": int(row[COL_PP]),
            "Pay Period Start": str(row[COL_START].date()) if pd.notna(row[COL_START]) else "",
            "Pay Period End": str(row[COL_END].date()) if pd.notna(row[COL_END]) else "",
            "Working Hours": float(row[COL_WORK]),
            "Billable Hours": float(row[COL_BILL]),
        }
        for _, row in dff.iterrows()
    ]

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div(
                    children=[
                        html.P("CBU Tracker", className="hero-eyebrow"),
                        html.H2("Dashboard Overview", className="hero-title"),
                        html.P(
                            "Track billable hours, monitor progress, and forecast fiscal-year performance.",
                            className="hero-subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="controls-card",
                    children=[
                        html.Div(
                            className="control-field",
                            children=[
                                html.Label("Fiscal Year", className="control-label"),
                                dcc.Dropdown(
                                    id="fy",
                                    options=[{"label": f"FY{y}", "value": y} for y in fy_values],
                                    value=fy_values[-1] if fy_values else None,
                                    clearable=False,
                                    className="control-dropdown",
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-field control-field-wide",
                            children=[
                                html.Label("CBU Goal (EOY Target: 90.8%)", className="control-label"),
                                dcc.Slider(
                                    id="goal",
                                    min=0.50,
                                    max=1.00,
                                    step=0.001,
                                    value=0.908,
                                    marks={0.50: "50%", 0.70: "70%", 0.908: "90.8%", 1.00: "100%"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="section-card",
            children=[
                html.Div(
                    className="section-header",
                    children=[
                        html.H3("Upload Billable Hours File"),
                        html.P(
                            "Upload an Excel (.xlsx) or CSV file with your billable hours. The file should have columns for "
                            "'PP#' (or 'Pay Period') and 'Billable Hours'.",
                        ),
                    ],
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(
                        [
                            "Drag and Drop or ",
                            html.A("Select a File"),
                        ]
                    ),
                    className="upload-dropzone",
                    multiple=False,
                ),
                html.Div(id="upload-status", className="upload-status"),
                html.Details(
                    [
                        html.Summary("Expected File Format"),
                        html.Div(
                            [
                                html.P("Supported formats:"),
                                html.P("✅ Technomics CBU Tracker Excel files (auto-detected from 'CBU Tracker' sheet)"),
                                html.P("✅ Simple Excel/CSV files with the following columns:"),
                                html.Ul(
                                    [
                                        html.Li("'PP#' or 'Pay Period' — pay period numbers (1-26)"),
                                        html.Li("'Billable Hours' — your billable hours for each period"),
                                        html.Li("'Working Hours' (optional) — available working hours"),
                                        html.Li("'Y' or 'FY' (optional) — fiscal year"),
                                    ]
                                ),
                            ],
                            className="details-content",
                        ),
                    ],
                    className="details-box",
                ),
            ],
        ),
        html.Div(
            className="section-card",
            children=[
                html.Div(
                    className="section-header",
                    children=[
                        html.H3("Enter Your Billable Hours"),
                        html.P("Edit the 'Billable Hours' column below, or upload a file above to auto-fill:"),
                    ],
                ),
                dash_table.DataTable(
                    id="input_table",
                    columns=[
                        {"name": "PP#", "id": "PP#", "editable": False},
                        {"name": "Pay Period Start", "id": "Pay Period Start", "editable": False},
                        {"name": "Pay Period End", "id": "Pay Period End", "editable": False},
                        {"name": "Working Hours", "id": "Working Hours", "editable": False},
                        {"name": "Billable Hours", "id": "Billable Hours", "editable": True, "type": "numeric"},
                    ],
                    data=get_initial_billable_data(fy_values[-1]) if fy_values else [],
                    editable=True,
                    row_deletable=False,
                    style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
                    style_cell={"padding": "8px", "fontSize": "13px", "textAlign": "center"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#1f3b73", "color": "white"},
                    style_data_conditional=[
                        {
                            "if": {"column_id": "Billable Hours"},
                            "backgroundColor": "#eef3ff",
                            "fontWeight": "bold",
                        }
                    ],
                ),
            ],
        ),
        html.Div(
            className="kpi-grid",
            children=[
                html.Div(id="kpi_cbu_ytd"),
                html.Div(id="kpi_cbu_period"),
                html.Div(id="kpi_bill_ytd"),
                html.Div(id="kpi_work_ytd"),
            ],
        ),
        html.Div(
            className="section-card",
            children=[
                dcc.Graph(id="cbu_line"),
                dcc.Graph(id="hours_bar"),
            ],
        ),
        html.Div(
            className="section-card",
            children=[
                html.Div(className="section-header", children=[html.H3("Pay Period Detail")]),
                dash_table.DataTable(
                    id="detail_table",
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "6px", "fontSize": "13px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#1f3b73", "color": "white"},
                ),
            ],
        ),
    ],
)


def kpi_card(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
            html.Div(subtitle, className="kpi-subtitle"),
        ],
    )


def parse_uploaded_file(contents: str, filename: str) -> tuple[pd.DataFrame | None, str]:
    """
    Parse an uploaded Excel or CSV file and extract billable hours data.
    Returns (dataframe, status_message).
    """
    if contents is None:
        return None, ""
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Determine file type and read accordingly
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            # Try to read the 'CBU Tracker' sheet first (Technomics format)
            try:
                # Read without header to find the header row
                df_raw = pd.read_excel(io.BytesIO(decoded), engine='openpyxl', 
                                       sheet_name='CBU Tracker', header=None)
                
                # Find the header row by looking for 'PP#' in any cell
                header_row = None
                for idx, row in df_raw.iterrows():
                    if 'PP#' in row.values:
                        header_row = idx
                        break
                
                if header_row is not None:
                    # Re-read with the correct header
                    df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl',
                                       sheet_name='CBU Tracker', header=header_row)
                else:
                    # Fallback to first sheet with default header
                    df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
            except Exception:
                # Sheet not found, try first sheet
                df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, f"❌ Unsupported file type: {filename}. Please upload .xlsx, .xls, or .csv"
        
        # Normalize column names (strip whitespace, keep original case for matching)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Create lowercase mapping for flexible column detection
        col_map = {str(c).lower(): c for c in df.columns}
        
        # Find pay period column
        pp_col = None
        for col_lower in ['pp#', 'pp', 'pay period', 'period', 'payperiod']:
            if col_lower in col_map:
                pp_col = col_map[col_lower]
                break
        
        if pp_col is None:
            return None, "❌ Could not find pay period column. Expected: 'PP#', 'PP', 'Pay Period', or 'Period'"
        
        # Find billable hours column
        bill_col = None
        for col_lower in ['billable hours', 'billable', 'hours', 'bill hours', 'billhours']:
            if col_lower in col_map:
                bill_col = col_map[col_lower]
                break
        
        if bill_col is None:
            return None, "❌ Could not find billable hours column. Expected: 'Billable Hours', 'Billable', or 'Hours'"
        
        # Extract relevant columns
        result = pd.DataFrame({
            'PP#': pd.to_numeric(df[pp_col], errors='coerce'),
            'Billable Hours': pd.to_numeric(df[bill_col], errors='coerce').fillna(0)
        })
        
        # Check for FY column (Y or FY or Fiscal Year)
        fy_col = None
        for col_lower in ['y', 'fy', 'fiscal year', 'fiscalyear', 'year']:
            if col_lower in col_map:
                fy_col = col_map[col_lower]
                break
        
        if fy_col:
            result['FY'] = pd.to_numeric(df[fy_col], errors='coerce')
        
        # Check for Working Hours column (to override defaults)
        work_col = None
        for col_lower in ['working hours', 'workinghours', 'work hours', 'available hours']:
            if col_lower in col_map:
                work_col = col_map[col_lower]
                break
        
        if work_col:
            result['Working Hours'] = pd.to_numeric(df[work_col], errors='coerce')
        
        # Remove rows with invalid PP# or rows that are notes/footers
        result = result.dropna(subset=['PP#'])
        result = result[result['PP#'].between(1, 26)]  # Valid pay periods only
        result['PP#'] = result['PP#'].astype(int)
        
        if len(result) == 0:
            return None, "❌ No valid data rows found in the file."
        
        return result, f"✅ Successfully loaded {len(result)} pay periods from '{filename}'"
        
    except Exception as e:
        return None, f"❌ Error parsing file: {str(e)}"


# Callback to handle file upload and update input table
@app.callback(
    Output("input_table", "data"),
    Output("upload-status", "children"),
    Input("fy", "value"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_input_table(fy: int, contents: str, filename: str):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    # Start with base data for the fiscal year
    base_data = get_initial_billable_data(fy)
    
    if triggered_id == "upload-data" and contents is not None:
        # Parse the uploaded file
        uploaded_df, status_msg = parse_uploaded_file(contents, filename)
        
        if uploaded_df is not None:
            # If file has FY column, filter to selected FY
            if 'FY' in uploaded_df.columns:
                uploaded_df = uploaded_df[uploaded_df['FY'] == fy]
                if len(uploaded_df) == 0:
                    return base_data, f"⚠️ File loaded but no data found for FY{fy}. Showing empty template."
            
            # Update base_data with uploaded billable hours (and working hours if available)
            uploaded_hours = dict(zip(uploaded_df['PP#'], uploaded_df['Billable Hours']))
            uploaded_work = dict(zip(uploaded_df['PP#'], uploaded_df['Working Hours'])) if 'Working Hours' in uploaded_df.columns else {}
            
            for row in base_data:
                pp = row['PP#']
                if pp in uploaded_hours:
                    row['Billable Hours'] = float(uploaded_hours[pp])
                if pp in uploaded_work and pd.notna(uploaded_work.get(pp)):
                    row['Working Hours'] = float(uploaded_work[pp])
            
            return base_data, status_msg
        else:
            return base_data, status_msg
    
    # Fiscal year changed or initial load - return base data
    return base_data, ""


@app.callback(
    Output("kpi_cbu_ytd", "children"),
    Output("kpi_cbu_period", "children"),
    Output("kpi_bill_ytd", "children"),
    Output("kpi_work_ytd", "children"),
    Output("cbu_line", "figure"),
    Output("hours_bar", "figure"),
    Output("detail_table", "columns"),
    Output("detail_table", "data"),
    Input("fy", "value"),
    Input("goal", "value"),
    Input("input_table", "data"),
)
def update(fy: int, goal: float, input_data: list):
    # Build dataframe from user input
    dff = df_all[df_all["FY"] == fy].copy()
    dff = dff.sort_values(COL_PP).reset_index(drop=True)
    
    # Override billable hours and working hours with user-entered values
    if input_data:
        user_billable = {row["PP#"]: row.get("Billable Hours", 0) or 0 for row in input_data}
        user_working = {row["PP#"]: row.get("Working Hours") for row in input_data}
        dff[COL_BILL] = dff[COL_PP].map(lambda pp: user_billable.get(int(pp), 0) if pd.notna(pp) else 0)
        # Update working hours if provided
        dff[COL_WORK] = dff[COL_PP].map(lambda pp: user_working.get(int(pp)) if pd.notna(pp) and user_working.get(int(pp)) else dff.loc[dff[COL_PP] == pp, COL_WORK].iloc[0] if len(dff.loc[dff[COL_PP] == pp]) > 0 else 80)

    # Compute CBU with no adjustment (user provides actual values now)
    calc = compute_cbu(dff, adj_billable_per_period=0)

    # KPIs: "current" = last pay period in FY
    last = calc.iloc[-1] if len(calc) else None

    cbu_ytd = float(last["CBU (YTD)"]) if last is not None and pd.notna(last["CBU (YTD)"]) else np.nan
    cbu_period = float(last["CBU (Period)"]) if last is not None and pd.notna(last["CBU (Period)"]) else np.nan
    bill_ytd = float(last["Billable (Adj) YTD"]) if last is not None else 0.0
    work_ytd = float(last["Working YTD"]) if last is not None else 0.0

    k1 = kpi_card("CBU (YTD)", f"{cbu_ytd:.3f}" if pd.notna(cbu_ytd) else "—", f"vs goal {goal:.2f}  (Δ {cbu_ytd-goal:+.3f})" if pd.notna(cbu_ytd) else "")
    k2 = kpi_card("CBU (Period) — latest PP", f"{cbu_period:.3f}" if pd.notna(cbu_period) else "—", "Based on your entered hours")
    k3 = kpi_card("Billable Hours YTD", f"{bill_ytd:,.1f}", "")
    k4 = kpi_card("Working Hours YTD", f"{work_ytd:,.1f}", "")

    # Line chart: Period vs YTD + goal line
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=calc[COL_PP].astype(int),
        y=calc["CBU (Period)"],
        mode="lines+markers",
        name="CBU (Period)"
    ))
    fig_line.add_trace(go.Scatter(
        x=calc[COL_PP].astype(int),
        y=calc["CBU (YTD)"],
        mode="lines+markers",
        name="CBU (YTD)"
    ))
    fig_line.add_trace(go.Scatter(
        x=calc[COL_PP].astype(int),
        y=[goal] * len(calc),
        mode="lines",
        name="Goal",
        line={"dash": "dash"}
    ))
    fig_line.update_layout(
        title=f"FY{fy} — CBU (Period) and CBU (YTD)",
        xaxis_title="Pay Period (PP#)",
        yaxis_title="CBU",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Bar chart: hours
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=calc[COL_PP].astype(int),
        y=calc[COL_WORK],
        name="Working Hours"
    ))
    fig_bar.add_trace(go.Bar(
        x=calc[COL_PP].astype(int),
        y=calc["Billable Hours (Adj)"],
        name="Billable Hours (Adj)"
    ))
    fig_bar.update_layout(
        barmode="group",
        title=f"FY{fy} — Hours by Pay Period",
        xaxis_title="Pay Period (PP#)",
        yaxis_title="Hours",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Table
    show_cols = [
        COL_PP, COL_START, COL_END,
        COL_WORK, COL_BILL,
        "Billable Hours (Adj)", "CBU (Period)", "CBU (YTD)"
    ]
    # Only include cols that exist (in case start/end missing)
    show_cols = [c for c in show_cols if c in calc.columns]

    tbl = calc[show_cols].copy()

    # Formatting for display
    for c in [COL_WORK, COL_BILL, "Billable Hours (Adj)"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].astype(float).round(1)

    for c in ["CBU (Period)", "CBU (YTD)"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].astype(float).round(4)

    for c in [COL_START, COL_END]:
        if c in tbl.columns:
            tbl[c] = pd.to_datetime(tbl[c]).dt.date.astype(str)

    columns = [{"name": c, "id": c} for c in tbl.columns]
    data = tbl.to_dict("records")

    return k1, k2, k3, k4, fig_line, fig_bar, columns, data


if __name__ == "__main__":
    app.run(debug=True)
