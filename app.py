# app.py
# Plotly Dash "CBU Tracker" app with What-If parameters
#
# Run:
#   pip install dash pandas plotly
#   python app.py
#
# Then open: http://127.0.0.1:8050

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
    FY runs from October 1 of the prior year to September 30.
    There are 26 pay periods per year (bi-weekly).
    """
    # FY26 starts Oct 1, 2025 and ends Sep 30, 2026
    fy_start = datetime(fy - 1, 10, 1)  # Oct 1 of prior calendar year
    
    pay_periods = []
    current_start = fy_start
    
    for pp in range(1, 27):  # 26 pay periods
        pp_end = current_start + timedelta(days=13)  # 14-day pay period
        
        # Standard working hours per pay period (80 hours = 2 weeks x 40 hours)
        working_hours = 80.0
        
        # Adjust for federal holidays (approximate)
        # PP1 (early Oct) - Columbus Day
        # PP5-6 (Nov) - Veterans Day, Thanksgiving
        # PP7-8 (Dec) - Christmas
        # PP9 (Jan) - New Year, MLK Day
        # PP12 (Feb) - Presidents Day
        # PP17 (May) - Memorial Day
        # PP19 (Jul) - Independence Day
        # PP21 (Sep) - Labor Day
        holiday_pps = {1: 72, 5: 72, 6: 64, 7: 72, 8: 72, 9: 64, 12: 72, 17: 72, 19: 72, 21: 72}
        working_hours = holiday_pps.get(pp, 80.0)
        
        pay_periods.append({
            COL_PP: pp,
            COL_START: current_start,
            COL_END: pp_end,
            COL_WORK: working_hours,
            COL_BILL: 0.0,  # User will enter this
        })
        
        current_start = pp_end + timedelta(days=1)
    
    return pd.DataFrame(pay_periods)


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
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "20px auto", "padding": "0 20px"},
    children=[
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
            ],
        ),

        html.Hr(style={"borderColor": "#00a0b2", "opacity": "0.3"}),

        html.H3("Enter Your Billable Hours", style={"color": "a"}),
        html.P("Edit the 'Billable Hours' column below to enter your hours for each pay period:", 
               style={"color": "#666", "fontSize": "14px"}),
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
                {
                    "if": {"column_id": "Billable Hours"},
                    "backgroundColor": "#fffde7",
                    "fontWeight": "bold",
                }
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
        style={
            "border": "2px solid #00a0b2",
            "borderRadius": "10px",
            "padding": "12px 14px",
            "boxShadow": "0 2px 4px rgba(0,160,178,0.15)",
            "background": "white",
        },
        children=[
            html.Div(title, style={"fontSize": "13px", "color": "#00a0b2", "fontWeight": "bold"}),
            html.Div(value, style={"fontSize": "30px", "fontWeight": "bold", "marginTop": "6px", "color": "#333"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "#777", "marginTop": "4px"}),
        ],
    )


# Callback to update input table when fiscal year changes
@app.callback(
    Output("input_table", "data"),
    Input("fy", "value"),
)
def update_input_table(fy: int):
    return get_initial_billable_data(fy)


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
    
    # Override billable hours with user-entered values
    if input_data:
        user_billable = {row["PP#"]: row.get("Billable Hours", 0) or 0 for row in input_data}
        dff[COL_BILL] = dff[COL_PP].map(lambda pp: user_billable.get(int(pp), 0) if pd.notna(pp) else 0)

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
