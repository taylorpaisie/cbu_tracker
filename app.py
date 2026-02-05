# app.py
# Plotly Dash "CBU Tracker" app with What-If parameters
#
# Run:
#   pip install dash pandas plotly openpyxl
#   python app.py
#
# Then open: http://127.0.0.1:8050

from __future__ import annotations

import base64
import io
import calendar
from datetime import date

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go

# ---- CONFIG ----
# Column names used internally
COL_PP = "PP#"
COL_FY = "FY"  # fiscal year label (e.g., 2026 for Apr 2025–Mar 2026)
COL_START = "Pay Period Start"
COL_END = "Pay Period End"
COL_WORK = "Working Hours"
COL_BILL = "Billable Hours"


def business_days_inclusive(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Mon–Fri business days inclusive (matches Technomics tracker working-hours behavior)."""
    return len(pd.date_range(start, end, freq="B"))


def generate_fy_pay_periods(fy: int) -> pd.DataFrame:
    """
    Generate pay period data for a fiscal year in the SAME format as the Technomics CBU Tracker workbook:

      FY<fy> runs Apr 1 of (fy-1) through Mar 31 of (fy)
      Pay periods are SEMI-MONTHLY (24 per FY):
        PP1  = Apr 1–15
        PP2  = Apr 16–end of month
        ...
        PP23 = Mar 1–15
        PP24 = Mar 16–end of month

      Working Hours = (business days in the pay period) * 8

    Example:
      FY26 => Apr 1, 2025 – Mar 31, 2026
    """
    fy_start = pd.Timestamp(date(fy - 1, 4, 1))
    fy_end = pd.Timestamp(date(fy, 3, 31))

    pay_periods = []
    pp = 1

    # Walk month-by-month from start to end
    cur = fy_start.normalize()
    while cur <= fy_end:
        y, m = int(cur.year), int(cur.month)
        last_dom = calendar.monthrange(y, m)[1]

        # Two semi-monthly periods
        p1_start = pd.Timestamp(date(y, m, 1))
        p1_end = pd.Timestamp(date(y, m, 15))

        p2_start = pd.Timestamp(date(y, m, 16))
        p2_end = pd.Timestamp(date(y, m, last_dom))

        for s, e in [(p1_start, p1_end), (p2_start, p2_end)]:
            # Clip to FY bounds
            if s < fy_start:
                s = fy_start
            if e > fy_end:
                e = fy_end

            if s <= e:
                working_hours = float(business_days_inclusive(s, e) * 8)

                pay_periods.append(
                    {
                        COL_PP: pp,
                        COL_FY: fy,
                        COL_START: s,
                        COL_END: e,
                        COL_WORK: working_hours,
                        COL_BILL: 0.0,  # user enters this (or upload)
                    }
                )
                pp += 1

        # Advance to first day of next month
        if m == 12:
            cur = pd.Timestamp(date(y + 1, 1, 1))
        else:
            cur = pd.Timestamp(date(y, m + 1, 1))

    return pd.DataFrame(pay_periods)


def generate_sample_data() -> pd.DataFrame:
    """Generate pay period templates for multiple fiscal years."""
    fiscal_years = [2025, 2026]  # FY25 (Apr 2024–Mar 2025) and FY26 (Apr 2025–Mar 2026)
    all_data = [generate_fy_pay_periods(fy) for fy in fiscal_years]
    return pd.concat(all_data, ignore_index=True)


def compute_cbu(df_fy: pd.DataFrame) -> pd.DataFrame:
    """Compute period and YTD CBU using Billable Hours / Working Hours."""
    out = df_fy.copy()

    out["Billable Hours (Adj)"] = out[COL_BILL].astype(float).clip(lower=0)

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

fy_values = sorted([int(x) for x in df_all[COL_FY].dropna().unique().tolist()])


# ---- Dash app ----
app = Dash(__name__)
app.title = "CBU Tracker"

# Expose the Flask server for gunicorn
server = app.server


def get_initial_billable_data(fy: int) -> list[dict]:
    """Get pay periods for a fiscal year with editable billable hours."""
    dff = df_all[df_all[COL_FY] == fy].copy().sort_values(COL_PP).reset_index(drop=True)
    return [
        {
            "PP#": int(row[COL_PP]),
            "Pay Period Start": str(pd.to_datetime(row[COL_START]).date()) if pd.notna(row[COL_START]) else "",
            "Pay Period End": str(pd.to_datetime(row[COL_END]).date()) if pd.notna(row[COL_END]) else "",
            "Working Hours": float(row[COL_WORK]) if pd.notna(row[COL_WORK]) else 0.0,
            "Billable Hours": float(row[COL_BILL]) if pd.notna(row[COL_BILL]) else 0.0,
        }
        for _, row in dff.iterrows()
    ]


app.layout = html.Div(
    id="app-container",
    className="theme-light",
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "20px auto", "padding": "0 20px"},
    children=[
        dcc.Store(id="theme-store", storage_type="local", data="light"),
        html.H2("CBU Tracker Dashboard", style={"color": "#00a0b2"}),
        html.Div(
            style={"display": "flex", "gap": "18px", "flexWrap": "wrap", "alignItems": "flex-end"},
            children=[
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Label("Fiscal Year", style={"color": "#00a0b2", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="fy",
                            options=[{"label": f"FY{y}", "value": y} for y in fy_values],
                            value=fy_values[-1] if fy_values else None,
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "320px", "flex": "1"},
                    children=[
                        html.Label("CBU Goal (EOY Target: 90.8%)", style={"color": "#00a0b2", "fontWeight": "bold"}),
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
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Label("Theme", style={"color": "#00a0b2", "fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="theme-toggle",
                            options=[
                                {"label": "Light", "value": "light"},
                                {"label": "Dark", "value": "dark"},
                            ],
                            value="light",
                            inline=True,
                            className="theme-toggle",
                            inputStyle={"marginRight": "6px", "marginLeft": "12px"},
                        ),
                    ],
                ),
            ],
        ),
        html.Hr(style={"borderColor": "#00a0b2", "opacity": "0.3"}),
        html.H3("Upload Billable Hours File", style={"color": "#00a0b2"}),
        html.P(
            "Upload an Excel (.xlsx) or CSV file with your billable hours. The file should have columns for "
            "'PP#' (or 'Pay Period') and 'Billable Hours'.",
            style={"color": "#666", "fontSize": "14px"},
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                [
                    "Drag and Drop or ",
                    html.A("Select a File", style={"color": "#00a0b2", "fontWeight": "bold", "cursor": "pointer"}),
                ]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "borderColor": "#00a0b2",
                "textAlign": "center",
                "marginBottom": "10px",
                "backgroundColor": "#f9fffe",
            },
            multiple=False,
        ),
        html.Div(id="upload-status", style={"marginBottom": "15px", "color": "#666", "fontSize": "14px"}),
        html.Details(
            [
                html.Summary("Expected File Format", style={"cursor": "pointer", "color": "#00a0b2", "fontWeight": "bold"}),
                html.Div(
                    [
                        html.P("Supported formats:", style={"marginTop": "10px", "fontWeight": "bold"}),
                        html.P("✅ Technomics CBU Tracker Excel files (auto-detected from 'CBU Tracker' sheet)"),
                        html.P("✅ Simple Excel/CSV files with the following columns:"),
                        html.Ul(
                            [
                                html.Li("'PP#' or 'Pay Period' — pay period numbers (1–24)"),
                                html.Li("'Billable Hours' — your billable hours for each period"),
                                html.Li("'Working Hours' (optional) — available working hours"),
                                html.Li("'FY' (optional) — fiscal year label (e.g., 2026)"),
                            ]
                        ),
                    ],
                    style={"padding": "10px", "backgroundColor": "#f5f5f5", "borderRadius": "5px", "marginTop": "5px"},
                ),
            ],
            style={"marginBottom": "15px"},
        ),
        html.Hr(style={"borderColor": "#00a0b2", "opacity": "0.3"}),
        html.H3("Enter Your Billable Hours", style={"color": "#00a0b2"}),
        html.P("Edit the 'Billable Hours' column below, or upload a file above to auto-fill:", style={"color": "#666", "fontSize": "14px"}),
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
            style_header={"fontWeight": "bold", "backgroundColor": "#00a0b2", "color": "white"},
            style_data_conditional=[
                {"if": {"column_id": "Billable Hours"}, "backgroundColor": "#fffde7", "fontWeight": "bold"}
            ],
        ),
        html.Hr(style={"borderColor": "#00a0b2", "opacity": "0.3"}),
        html.Div(
            style={"display": "flex", "gap": "18px", "flexWrap": "wrap"},
            children=[
                html.Div(id="kpi_cbu_ytd", style={"flex": "1", "minWidth": "240px"}),
                html.Div(id="kpi_cbu_period", style={"flex": "1", "minWidth": "240px"}),
                html.Div(id="kpi_bill_ytd", style={"flex": "1", "minWidth": "240px"}),
                html.Div(id="kpi_work_ytd", style={"flex": "1", "minWidth": "240px"}),
            ],
        ),
        html.Hr(style={"borderColor": "#00a0b2", "opacity": "0.3"}),
        dcc.Graph(id="cbu_line"),
        dcc.Graph(id="hours_bar"),
        html.H3("Pay Period Detail", style={"color": "#00a0b2"}),
        dash_table.DataTable(
            id="detail_table",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#00a0b2", "color": "white"},
        ),
    ],
)


def kpi_card(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        className="kpi-card",
        style={
            "border": "2px solid var(--accent)",
            "borderRadius": "10px",
            "padding": "12px 14px",
            "boxShadow": "0 2px 4px var(--shadow)",
            "background": "var(--card-bg)",
        },
        children=[
            html.Div(title, style={"fontSize": "13px", "color": "var(--accent)", "fontWeight": "bold"}),
            html.Div(value, style={"fontSize": "30px", "fontWeight": "bold", "marginTop": "6px", "color": "var(--text-main)"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "4px"}),
        ],
    )


@app.callback(
    Output("theme-store", "data"),
    Output("theme-toggle", "value"),
    Output("app-container", "className"),
    Input("theme-toggle", "value"),
    State("theme-store", "data"),
)
def sync_theme(theme_value: str, stored_theme: str):
    selected = theme_value or stored_theme or "light"
    return selected, selected, f"theme-{selected}"


def parse_uploaded_file(contents: str, filename: str) -> tuple[pd.DataFrame | None, str]:
    """
    Parse an uploaded Excel or CSV file and extract billable hours data.
    Returns (dataframe, status_message).
    """
    if contents is None:
        return None, ""

    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Determine file type and read accordingly
        if filename.lower().endswith((".xlsx", ".xls")):
            # Try Technomics format first
            try:
                df_raw = pd.read_excel(io.BytesIO(decoded), engine="openpyxl", sheet_name="CBU Tracker", header=None)

                header_row = None
                for idx, row in df_raw.iterrows():
                    if any(str(v).strip() == "PP#" for v in row.values):
                        header_row = idx
                        break

                if header_row is not None:
                    df = pd.read_excel(
                        io.BytesIO(decoded), engine="openpyxl", sheet_name="CBU Tracker", header=header_row
                    )
                else:
                    df = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
            except Exception:
                df = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")

        elif filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            return None, f"❌ Unsupported file type: {filename}. Please upload .xlsx, .xls, or .csv"

        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        col_map = {str(c).lower(): c for c in df.columns}

        # Find pay period column
        pp_col = None
        for col_lower in ["pp#", "pp", "pay period", "period", "payperiod"]:
            if col_lower in col_map:
                pp_col = col_map[col_lower]
                break
        if pp_col is None:
            return None, "❌ Could not find pay period column. Expected: 'PP#', 'PP', 'Pay Period', or 'Period'"

        # Find billable hours column
        bill_col = None
        for col_lower in ["billable hours", "billable", "hours", "bill hours", "billhours"]:
            if col_lower in col_map:
                bill_col = col_map[col_lower]
                break
        if bill_col is None:
            return None, "❌ Could not find billable hours column. Expected: 'Billable Hours', 'Billable', or 'Hours'"

        result = pd.DataFrame(
            {
                "PP#": pd.to_numeric(df[pp_col], errors="coerce"),
                "Billable Hours": pd.to_numeric(df[bill_col], errors="coerce").fillna(0),
            }
        )

        # Optional FY column (prefer FY label, but accept Y if someone uses it as FY)
        fy_col = None
        for col_lower in ["fy", "fiscal year", "fiscalyear", "year", "y"]:
            if col_lower in col_map:
                fy_col = col_map[col_lower]
                break
        if fy_col:
            result["FY"] = pd.to_numeric(df[fy_col], errors="coerce")

        # Optional Working Hours override
        work_col = None
        for col_lower in ["working hours", "workinghours", "work hours", "available hours"]:
            if col_lower in col_map:
                work_col = col_map[col_lower]
                break
        if work_col:
            result["Working Hours"] = pd.to_numeric(df[work_col], errors="coerce")

        # Clean rows
        result = result.dropna(subset=["PP#"])
        result = result[result["PP#"].between(1, 24)]
        result["PP#"] = result["PP#"].astype(int)

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

    base_data = get_initial_billable_data(fy)

    if triggered_id == "upload-data" and contents is not None:
        uploaded_df, status_msg = parse_uploaded_file(contents, filename)

        if uploaded_df is not None:
            # If file has FY column, filter to selected FY
            if "FY" in uploaded_df.columns and uploaded_df["FY"].notna().any():
                uploaded_df = uploaded_df[uploaded_df["FY"] == fy]
                if len(uploaded_df) == 0:
                    return base_data, f"⚠️ File loaded but no data found for FY{fy}. Showing template."

            uploaded_hours = dict(zip(uploaded_df["PP#"], uploaded_df["Billable Hours"]))
            uploaded_work = (
                dict(zip(uploaded_df["PP#"], uploaded_df["Working Hours"])) if "Working Hours" in uploaded_df.columns else {}
            )

            for row in base_data:
                pp = row["PP#"]
                if pp in uploaded_hours:
                    row["Billable Hours"] = float(uploaded_hours[pp])
                if pp in uploaded_work and pd.notna(uploaded_work.get(pp)):
                    row["Working Hours"] = float(uploaded_work[pp])

            return base_data, status_msg

        return base_data, status_msg

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
    Input("theme-store", "data"),
)
def update(fy: int, goal: float, input_data: list, theme: str):
    dff = df_all[df_all[COL_FY] == fy].copy().sort_values(COL_PP).reset_index(drop=True)

    # Override billable hours / working hours with user-entered values
    if input_data:
        user_billable = {int(row["PP#"]): float(row.get("Billable Hours", 0) or 0) for row in input_data if row.get("PP#") is not None}
        user_working = {int(row["PP#"]): row.get("Working Hours") for row in input_data if row.get("PP#") is not None}

        dff[COL_BILL] = dff[COL_PP].astype(int).map(lambda pp: user_billable.get(pp, 0.0))

        def pick_work(pp: int, default_val: float) -> float:
            v = user_working.get(pp, None)
            if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
                return float(default_val)
            try:
                return float(v)
            except Exception:
                return float(default_val)

        dff[COL_WORK] = [
            pick_work(int(pp), float(default_wh))
            for pp, default_wh in zip(dff[COL_PP].astype(int).tolist(), dff[COL_WORK].astype(float).tolist())
        ]

    calc = compute_cbu(dff)

    last = calc.iloc[-1] if len(calc) else None
    cbu_ytd = float(last["CBU (YTD)"]) if last is not None and pd.notna(last["CBU (YTD)"]) else np.nan
    cbu_period = float(last["CBU (Period)"]) if last is not None and pd.notna(last["CBU (Period)"]) else np.nan
    bill_ytd = float(last["Billable (Adj) YTD"]) if last is not None else 0.0
    work_ytd = float(last["Working YTD"]) if last is not None else 0.0

    k1 = kpi_card(
        "CBU (YTD)",
        f"{cbu_ytd:.3f}" if pd.notna(cbu_ytd) else "—",
        f"vs goal {goal:.2f}  (Δ {cbu_ytd-goal:+.3f})" if pd.notna(cbu_ytd) else "",
    )
    k2 = kpi_card("CBU (Period) — latest PP", f"{cbu_period:.3f}" if pd.notna(cbu_period) else "—", "Based on your entered hours")
    k3 = kpi_card("Billable Hours YTD", f"{bill_ytd:,.1f}", "")
    k4 = kpi_card("Working Hours YTD", f"{work_ytd:,.1f}", "")

    # Line chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=calc[COL_PP].astype(int), y=calc["CBU (Period)"], mode="lines+markers", name="CBU (Period)"))
    fig_line.add_trace(go.Scatter(x=calc[COL_PP].astype(int), y=calc["CBU (YTD)"], mode="lines+markers", name="CBU (YTD)"))
    fig_line.add_trace(go.Scatter(x=calc[COL_PP].astype(int), y=[goal] * len(calc), mode="lines", name="Goal", line={"dash": "dash"}))
    is_dark = theme == "dark"
    fig_line.update_layout(
        title=f"FY{fy} — CBU (Period) and CBU (YTD)",
        xaxis_title="Pay Period (PP#)",
        yaxis_title="CBU",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=calc[COL_PP].astype(int), y=calc[COL_WORK], name="Working Hours"))
    fig_bar.add_trace(go.Bar(x=calc[COL_PP].astype(int), y=calc["Billable Hours (Adj)"], name="Billable Hours"))
    fig_bar.update_layout(
        barmode="group",
        title=f"FY{fy} — Hours by Pay Period",
        xaxis_title="Pay Period (PP#)",
        yaxis_title="Hours",
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Detail table
    show_cols = [COL_PP, COL_START, COL_END, COL_WORK, COL_BILL, "Billable Hours (Adj)", "CBU (Period)", "CBU (YTD)"]
    tbl = calc[show_cols].copy()

    for c in [COL_WORK, COL_BILL, "Billable Hours (Adj)"]:
        tbl[c] = tbl[c].astype(float).round(1)

    for c in ["CBU (Period)", "CBU (YTD)"]:
        tbl[c] = tbl[c].astype(float).round(4)

    for c in [COL_START, COL_END]:
        tbl[c] = pd.to_datetime(tbl[c]).dt.date.astype(str)

    columns = [{"name": c, "id": c} for c in tbl.columns]
    data = tbl.to_dict("records")

    return k1, k2, k3, k4, fig_line, fig_bar, columns, data


if __name__ == "__main__":
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)
