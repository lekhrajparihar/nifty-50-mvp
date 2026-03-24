import os
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


# -----------------------------
# Configuration (MVP / single file)
# -----------------------------
DB_PATH = Path(__file__).parent / "test_breadth.db"
TABLE_NAME = "nifty50_prices"

EMA_WINDOWS = [20, 50, 100, 200]
LOOKBACK_52W = 252  # ~252 trading days
TOL_1PCT = 0.01

# NIFTY 50 constituents as of 8 Dec 2025 (per Wikipedia)
NIFTY50_TICKERS = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDIGO.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "MAXHEALTH.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SHRIRAMFIN.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TCS.NS",
    "TATACONSUM.NS",
    "TMPV.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]


def _init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            ticker VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT
        )
        """
    )


def _download_ticker_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV for a single ticker and normalize columns.
    Returns empty DataFrame if no data.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns even for single ticker, e.g. ("Open", "RELIANCE.NS").
    # Flatten to the first level so OHLCV fields become 1D Series.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # yfinance uses "Date" for the index column.
    if "Date" not in df.columns:
        # Fallback (rare): attempt common variants.
        for col in ["index", "Datetime", "datetime"]:
            if col in df.columns:
                df = df.rename(columns={col: "Date"})
                break

    if "Date" not in df.columns:
        return pd.DataFrame()

    def _col_1d(frame: pd.DataFrame, col_name: str, fallback: str | None = None) -> pd.Series:
        val = frame.get(col_name)
        if val is None and fallback is not None:
            val = frame.get(fallback)
        if val is None:
            return pd.Series([pd.NA] * len(frame))
        if isinstance(val, pd.DataFrame):
            if val.shape[1] == 0:
                return pd.Series([pd.NA] * len(frame))
            return val.iloc[:, 0]
        return val

    # Normalize to our schema.
    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(df["Date"]).dt.date,
            "open": _col_1d(df, "Open"),
            "high": _col_1d(df, "High"),
            "low": _col_1d(df, "Low"),
            "close": _col_1d(df, "Close"),
            "adj_close": _col_1d(df, "Adj Close", fallback="Close"),
            "volume": _col_1d(df, "Volume"),
        }
    )

    out = out.dropna(subset=["close", "date"])
    return out


def _insert_prices_for_ticker(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    prices: pd.DataFrame,
    delete_start: date | None = None,
    delete_end: date | None = None,
) -> int:
    if prices.empty:
        return 0

    if delete_start is not None and delete_end is not None:
        con.execute(
            f"""
            DELETE FROM {TABLE_NAME}
            WHERE ticker = ? AND date >= ? AND date <= ?
            """,
            [ticker, delete_start, delete_end],
        )
    else:
        # For sync (single day), delete by exact (ticker, date).
        dates = sorted(set(prices["date"].tolist()))
        if dates:
            con.execute(
                f"DELETE FROM {TABLE_NAME} WHERE ticker = ? AND date IN ({','.join(['?'] * len(dates))})",
                [ticker, *dates],
            )

    con.register("price_df", prices)
    con.execute(
        f"""
        INSERT INTO {TABLE_NAME} (ticker, date, open, high, low, close, adj_close, volume)
        SELECT ticker, date, open, high, low, close, adj_close, volume
        FROM price_df
        """
    )
    return int(len(prices))


def ensure_initial_2y_data(con: duckdb.DuckDBPyConnection, tickers: list[str]) -> dict:
    """
    If DB/table is empty, download last ~2 years and insert.
    """
    _init_db(con)

    count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    if count and count > 0:
        return {"did_initial_load": False}

    today = date.today()
    start_dt = today - timedelta(days=730)
    end_dt = today + timedelta(days=1)

    inserted_total = 0
    failed: list[str] = []

    st.info("Downloading ~2 years of daily data for Nifty 50 (one-time). This can take a minute.")
    progress = st.progress(0, text="Downloading tickers...")

    for i, ticker in enumerate(tickers, start=1):
        df = _download_ticker_daily(ticker, start_dt.isoformat(), end_dt.isoformat())
        if df.empty:
            failed.append(ticker)
        else:
            inserted_total += _insert_prices_for_ticker(
                con,
                ticker=ticker,
                prices=df,
                delete_start=start_dt,
                delete_end=end_dt,
            )
        progress.progress(i / len(tickers), text=f"Downloading tickers... ({i}/{len(tickers)})")

    progress.empty()
    st.success(f"Initial load complete. Inserted {inserted_total:,} rows.")
    if failed:
        st.warning(f"Some tickers returned no data and were skipped ({len(failed)}).")
    return {"did_initial_load": True, "inserted_total": inserted_total, "failed": failed}


def sync_latest_missing_day(con: duckdb.DuckDBPyConnection, tickers: list[str]) -> dict:
    """
    Append only the latest missing trading day per ticker:
    - For each ticker, find MAX(date) already stored.
    - Download from (max_date + 1 day) to today.
    - Insert only the latest available row beyond max_date.
    """
    today = date.today()
    inserted = 0
    skipped_no_new = 0
    had_no_history = 0

    progress = st.progress(0, text="Syncing latest missing day...")

    for i, ticker in enumerate(tickers, start=1):
        res = con.execute(f"SELECT MAX(date) FROM {TABLE_NAME} WHERE ticker = ?", [ticker]).fetchone()
        max_date = res[0] if res else None

        if max_date is None:
            had_no_history += 1
            continue

        start_dt = max_date + timedelta(days=1)
        end_dt = today + timedelta(days=1)

        df = _download_ticker_daily(ticker, start_dt.isoformat(), end_dt.isoformat())
        if df.empty:
            skipped_no_new += 1
        else:
            df = df[df["date"] > max_date]
            if df.empty:
                skipped_no_new += 1
            else:
                latest_row = df.sort_values("date").tail(1)
                inserted += _insert_prices_for_ticker(con, ticker, latest_row)

        progress.progress(i / len(tickers), text=f"Syncing latest missing day... ({i}/{len(tickers)})")

    progress.empty()
    return {
        "inserted_rows": inserted,
        "skipped_no_new": skipped_no_new,
        "tickers_without_history": had_no_history,
    }


def compute_breadth_and_summary(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    prices = con.execute(f"SELECT * FROM {TABLE_NAME}").df()
    if prices.empty:
        return pd.DataFrame(), {}

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    # Compute EMAs and breadth percentages.
    for w in EMA_WINDOWS:
        prices[f"ema{w}"] = prices.groupby("ticker")["close"].transform(
            lambda s: s.ewm(span=w, adjust=False, min_periods=w).mean()
        )

    # Above-EMA + valid-EMA counts for correct denominators.
    for w in EMA_WINDOWS:
        prices[f"ema{w}_valid"] = prices[f"ema{w}"].notna()
        prices[f"above_ema{w}"] = (prices["close"] > prices[f"ema{w}"]) & prices[f"ema{w}_valid"]

    grouped = prices.groupby("date", as_index=False)
    breadth_df = grouped.agg(
        **{
            **{f"valid{w}": (f"ema{w}_valid", "sum") for w in EMA_WINDOWS},
            **{f"above{w}": (f"above_ema{w}", "sum") for w in EMA_WINDOWS},
        }
    )
    for w in EMA_WINDOWS:
        breadth_df[f"breadth{w}"] = (
            breadth_df[f"above{w}"] / breadth_df[f"valid{w}"].replace({0: pd.NA})
        ) * 100.0

    keep_cols = ["date"] + [f"breadth{w}" for w in EMA_WINDOWS]
    breadth_df = breadth_df[keep_cols].sort_values("date").reset_index(drop=True)

    latest_date = breadth_df["date"].max()
    latest_slice = prices[prices["date"] == latest_date].copy()

    # 52-week high/low (approx. 252 trading days based on close).
    prices["high_52"] = prices.groupby("ticker")["close"].transform(
        lambda s: s.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W).max()
    )
    prices["low_52"] = prices.groupby("ticker")["close"].transform(
        lambda s: s.rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W).min()
    )
    prices["high_52_prev"] = prices.groupby("ticker")["high_52"].shift(1)
    prices["low_52_prev"] = prices.groupby("ticker")["low_52"].shift(1)

    latest_slice = prices[prices["date"] == latest_date].copy()
    latest_slice["new_52_high"] = (
        latest_slice["high_52_prev"].notna()
        & (latest_slice["high_52"] > latest_slice["high_52_prev"])
        & (latest_slice["close"] >= latest_slice["high_52"] * (1.0 - TOL_1PCT))
    )
    latest_slice["new_52_low"] = (
        latest_slice["low_52_prev"].notna()
        & (latest_slice["low_52"] < latest_slice["low_52_prev"])
        & (latest_slice["close"] <= latest_slice["low_52"] * (1.0 + TOL_1PCT))
    )

    high_count = int(latest_slice["new_52_high"].sum())
    low_count = int(latest_slice["new_52_low"].sum())

    latest_row = breadth_df[breadth_df["date"] == latest_date].iloc[0].to_dict()

    summary = {
        "latest_date": latest_date.date(),
        "breadth": {w: latest_row.get(f"breadth{w}", pd.NA) for w in EMA_WINDOWS},
        "new_52_week_high_count": high_count,
        "new_52_week_low_count": low_count,
    }
    return breadth_df, summary


def _fmt_pct(x) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{float(x):.1f}%"


def main() -> None:
    st.set_page_config(page_title="Nifty Market Breadth (EMA)", layout="wide")
    st.title("Nifty 50 Market Breadth Dashboard (MVP)")
    st.caption("DuckDB local storage + yfinance daily data + EMA breadth (% above EMA).")

    con = duckdb.connect(str(DB_PATH))
    _init_db(con)

    # Ensure initial load exists.
    with st.spinner("Checking local DuckDB price data..."):
        ensure_initial_2y_data(con, NIFTY50_TICKERS)

    db_last_date = con.execute(f"SELECT MAX(date) FROM {TABLE_NAME}").fetchone()[0]
    db_last_date = pd.to_datetime(db_last_date).date() if db_last_date is not None else None

    st.sidebar.header("Data")
    st.sidebar.markdown(f"**DB last date:** {db_last_date or 'N/A'}")

    if st.sidebar.button("Sync", use_container_width=True):
        with st.spinner("Syncing latest missing day..."):
            res = sync_latest_missing_day(con, NIFTY50_TICKERS)
        st.sidebar.success(f"Sync complete: +{res['inserted_rows']:,} rows inserted.")
        st.rerun()

    breadth_df, summary = compute_breadth_and_summary(con)
    if breadth_df.empty or not summary:
        st.warning("No data available to compute breadth.")
        return

    # Summary at top.
    st.subheader("Summary Metric")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Latest Date", str(summary["latest_date"]))
    c2.metric("Breadth > 20 EMA", _fmt_pct(summary["breadth"][20]))
    c3.metric("Breadth > 50 EMA", _fmt_pct(summary["breadth"][50]))
    c4.metric("Breadth > 100 EMA", _fmt_pct(summary["breadth"][100]))
    c5.metric("Breadth > 200 EMA", _fmt_pct(summary["breadth"][200]))
    c6.metric("New 52w High / Low", f'{summary["new_52_week_high_count"]} / {summary["new_52_week_low_count"]}')

    st.subheader("Breadth Over Time")
    chart_cols = st.columns(2)
    ema_to_plot = [(20, "breadth20"), (50, "breadth50"), (100, "breadth100"), (200, "breadth200")]

    for idx, (w, ycol) in enumerate(ema_to_plot):
        col = chart_cols[idx % 2]
        fig = px.line(
            breadth_df,
            x="date",
            y=ycol,
            title=f"Breadth (% Close > {w} EMA)",
        )
        fig.update_traces(line=dict(width=2))
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Date",
            yaxis_title="Breadth (%)",
        )
        col.plotly_chart(fig, use_container_width=True)

    # Optional: show which EMA windows are partially unavailable early in history.
    st.caption(
        "Note: EMA200 breadth may be `N/A` for early dates until enough history exists."
    )


if __name__ == "__main__":
    main()

