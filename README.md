# CBU Tracker

Interactive Plotly Dash app for tracking billable utilization (CBU) by semi-monthly pay period.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app starts on `http://127.0.0.1:8050` by default.

## Deploy to Posit Connect

This repository is now ready for Posit Connect deployment as a Dash app (`app.py` exposes `app` and `server`).

### 1) Prerequisites

- Access to a Posit Connect server
- Python installed locally (matching a version available on Connect)
- `rsconnect-python` installed:

```bash
pip install rsconnect-python
```

### 2) Authenticate to Connect

```bash
rsconnect add \
  --name my-connect \
  --server https://connect.example.com \
  --api-key <CONNECT_API_KEY>
```

### 3) Deploy

From the repository root:

```bash
rsconnect deploy dash \
  --name my-connect \
  --entrypoint app:app \
  .
```

Connect will install dependencies from `requirements.txt` and launch the app using the Dash entrypoint.

## Runtime configuration

When running directly (`python app.py`), the app supports:

- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8050`)
- `DASH_DEBUG` (default: `false`)

Example:

```bash
HOST=0.0.0.0 PORT=3939 DASH_DEBUG=true python app.py
```
