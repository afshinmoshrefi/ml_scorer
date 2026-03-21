"""Fetch earnings dates for all S&P 500 symbols from EDGAR API and cache locally."""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
API_BASE = "http://104.238.214.253:7670/earnings"


def fetch_one(symbol):
    """Fetch earnings dates for one symbol. Returns (symbol, list_of_8KE_dates)."""
    try:
        url = f"{API_BASE}/{symbol}"
        resp = urllib.request.urlopen(url, timeout=15)
        data = json.loads(resp.read())
        filings = data.get("filings", [])
        # Extract 8-K/E dates (actual earnings announcements)
        earnings = sorted(set(f["date"] for f in filings if f.get("form") == "8-K/E"))
        return symbol, earnings, None
    except Exception as e:
        return symbol, [], str(e)


def main():
    # Get symbol list from backtester input
    df = pd.read_parquet(RESULTS / "backtester_input_10_30.parquet", columns=["symbol"])
    symbols = sorted(df["symbol"].unique())
    print(f"Fetching earnings for {len(symbols)} symbols...")

    t0 = time.time()
    results = {}
    errors = []

    # Parallel fetch with 20 threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        done = 0
        for future in as_completed(futures):
            sym, dates, err = future.result()
            if err:
                errors.append((sym, err))
            results[sym] = dates
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(symbols)} fetched...")

    print(f"Fetched in {time.time()-t0:.1f}s")
    print(f"  Symbols with data: {sum(1 for v in results.values() if v)}")
    print(f"  Symbols without data: {sum(1 for v in results.values() if not v)}")
    if errors:
        print(f"  Errors: {len(errors)} (first 5: {errors[:5]})")

    # Coverage analysis
    have_2018 = sum(1 for v in results.values() if any(d.startswith("2018") for d in v))
    have_2020 = sum(1 for v in results.values() if any(d.startswith("2020") for d in v))
    have_2024 = sum(1 for v in results.values() if any(d.startswith("2024") for d in v))
    print(f"\n  Coverage: 2018={have_2018}, 2020={have_2020}, 2024={have_2024} symbols with earnings data")

    # Avg earnings per symbol
    counts = [len(v) for v in results.values() if v]
    if counts:
        print(f"  Avg earnings dates per symbol: {sum(counts)/len(counts):.1f}")
        print(f"  Min: {min(counts)}, Max: {max(counts)}")

    # Save
    out_path = RESULTS / "earnings_dates.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
