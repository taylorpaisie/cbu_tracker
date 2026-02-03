# app.py
# Plotly Dash "CBU Tracker" app with What-If parameters
#
# Run:
#   pip install dash pandas openpyxl plotly
#   python app.py
#
# Then open: http://127.0.0.1:8050

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go

# ---- CONFIG ----
EXCEL_PATH = r"Technomics CBU Tracker FY26 (1).xlsx"
SHEET_NAME = "CBU Tracker"  # change if needed

# Excel columns we expect
COL_PP = "PP#"
COL_FY = "Y"  # fiscal year column in your file
COL_START = "Pay Period Start"
COL_END = "Pay Period End"
COL_WORK = "Working Hours"
COL_BILL = "Billable Hours"
COL_BILL_YTD = "Billable Hours (YTD)"


def load_and_clean_tracker(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Loads the Excel sheet and finds the header row that contains 'PP#'.
    Then cleans types and returns a tidy dataframe.
    """
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")

    # Find the row index where any of the first few columns equals 'PP#' (header row)
    header_row_idx = None
    for i in range(min(len(raw), 50)):
        for j in range(min(len(raw.columns), 10)):
            v = raw.iloc[i, j]
            if isinstance(v, str) and v.strip() == COL_PP:
                header_row_idx = i
                break
        if header_row_idx is not None:
            break

    if header_row_idx is None:
        raise ValueError(
            "Couldn't find the header row (where a column is 'PP#'). "
            "Open the sheet and confirm the column header text."
        )

    df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=header_row_idx,
        engine="openpyxl",
    )

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Normalize column names (strip weird spaces/newlines)
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

    # Keep rows that actually have a pay period number
    df = df[df[COL_PP].notna()].copy()

    # Types
    df[COL_PP] = pd.to_numeric(df[COL_PP], errors="coerce").astype("Int64")
    if COL_FY in df.columns:
        df[COL_FY] = pd.to_numeric(df[COL_FY], errors="coerce").astype("Int64")

    for c in (COL_WORK, COL_BILL, COL_BILL_YTD):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in (COL_START, COL_END):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Sort properly
    sort_cols = [COL_FY, COL_PP] if COL_FY in df.columns else [COL_PP]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # If we only have YTD billable hours, derive per-period billable hours
    if COL_BILL not in df.columns and COL_BILL_YTD in df.columns:
        df[COL_BILL] = df[COL_BILL_YTD].diff().fillna(df[COL_BILL_YTD])
        # Handle negative values (new fiscal year reset)
        df.loc[df[COL_BILL] < 0, COL_BILL] = df.loc[df[COL_BILL] < 0, COL_BILL_YTD]

    # Some safety
    missing = [c for c in [COL_PP, COL_WORK, COL_BILL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}. Found: {list(df.columns)}")

    return df


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


# ---- Load data once at startup ----
df_all = load_and_clean_tracker(EXCEL_PATH, SHEET_NAME)

# Compute Fiscal Year as the ending year (Apr 2025 -> Mar 2026 = FY26)
# FY is based on the end date: if month >= 4, FY = year + 1; else FY = year
end_dt = df_all[COL_END].fillna(df_all[COL_START])
df_all["FY"] = end_dt.dt.year.where(end_dt.dt.month < 4, end_dt.dt.year + 1).astype("Int64")

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
