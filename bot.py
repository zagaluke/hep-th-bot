# bot.py


import sys
print(">>> Python executable:", sys.executable)
print(">>> sys.path[0]:", sys.path[0])

import os, re, requests, pandas as pd, numpy as np, joblib, tweepy
from dotenv import load_dotenv; load_dotenv()
from bs4 import BeautifulSoup
from features import build_features
import lightgbm as lgb

DATA_DIR   = os.getenv("DATA_DIR", "data")
DAILY_CSV  = os.path.join(DATA_DIR, "daily_counts.csv")
HOL_CSV    = os.path.join(DATA_DIR, "holidays.csv")
CONF_CSV   = os.path.join(DATA_DIR, "conferences.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "GBTmodel.pkl")
LOG_CSV    = os.path.join(DATA_DIR, "pred_log.csv")
TZ         = "Europe/Dublin"
UA         = "hepth-bot/0.1 (mailto:your.email@example.com)"


def load_any_lgb_model(path):
    # If you saved a native LightGBM text model (model.txt)
    if path.lower().endswith(".txt"):
        return lgb.Booster(model_file=path)
    # Else try joblib/pickle
    m = joblib.load(path)
    return m



def lgb_predict_any(model, X_frame):
    # sklearn-style estimator (LGBMRegressor / LGBMModel)
    if hasattr(model, "get_params") and hasattr(model, "predict"):
        return model.predict(X_frame)
    # raw Booster
    if isinstance(model, lgb.Booster):
        X = X_frame.values if hasattr(X_frame, "values") else X_frame
        return model.predict(X)
    # Fallback: try calling predict anyway
    return model.predict(X_frame)


FEATURE_COLS = [
    'days_until_public_holiday',
    'days_since_last_public_holiday',
    'day',
    'weekday',
    'days_since_start',
    'mean_same_weekday_4w',
    'days_since_last_conference',
    'roll_7',
    'roll_30',
    'roll_90',
    'roll_365',
    'lag_1',
    'lag_7',
    'days_until_next_conference',
]



def today_dub(): return pd.Timestamp.now(tz=TZ).normalize()
def dow_dub(): return int(today_dub().dayofweek)  # Mon=0..Sun=6
def load_csvs():
    dc  = pd.read_csv(DAILY_CSV, parse_dates=["date_first_appeared"]) if os.path.exists(DAILY_CSV) else pd.DataFrame(columns=["date_first_appeared","num_papers"])
    hol = pd.read_csv(HOL_CSV, parse_dates=["date"]) if os.path.exists(HOL_CSV) else pd.DataFrame(columns=["date"])
    conf= pd.read_csv(CONF_CSV, parse_dates=["start_date","end_date"]) if os.path.exists(CONF_CSV) else pd.DataFrame(columns=["start_date","end_date"])
    if len(dc):
        dc = dc.sort_values("date_first_appeared").reset_index(drop=True)
    return dc, hol, conf
def save_dc(df): df.sort_values("date_first_appeared").to_csv(DAILY_CSV, index=False)

def read_log():
    # ensure file exists
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=["date","yhat","tweeted_at","err_abs","actual"]).to_csv(LOG_CSV, index=False)
    log = pd.read_csv(LOG_CSV)
    # coerce to datetime (handles empty or bad rows)
    if "date" in log.columns:
        log["date"] = pd.to_datetime(log["date"], errors="coerce")
    if "tweeted_at" in log.columns:
        log["tweeted_at"] = pd.to_datetime(log["tweeted_at"], errors="coerce")
    return log



def write_log(date, yhat): 
    log = read_log()
    log = pd.concat([log, pd.DataFrame([{"date": pd.Timestamp(date), "yhat": float(yhat), "tweeted_at": pd.Timestamp.now(tz=TZ)}])], ignore_index=True)
    log.to_csv(LOG_CSV, index=False)

def fetch_hepth_new_count_current():
    r = requests.get("https://arxiv.org/list/hep-th/new?show=2000", headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    h3 = soup.find("h3", string=re.compile(r"^New submissions", re.I))
    if not h3: return 0
    m = re.search(r"showing\s+(\d+)\s+of\s+(\d+)\s+entries", h3.get_text(), re.I)
    if m: return int(m.group(2))
    cnt = 0
    for el in h3.find_all_next():
        if el.name == "h3": break
        if el.name == "dt": cnt += 1
    return cnt

def append_if_missing(dc, date_obj, count):
    ts = pd.Timestamp(date_obj)
    if (dc["date_first_appeared"] == ts).any(): return dc
    return pd.concat([dc, pd.DataFrame([{"date_first_appeared": ts, "num_papers": int(count)}])], ignore_index=True)

def update_daily_counts():
    # daily_counts.csv may already include the feature columns but only date/num_papers are filled;
    # we always rebuild all features and overwrite the file.
    dc, hol, conf = load_csvs()
    d = today_dub().date()
    dow = dow_dub()

    # work from counts only
    base = dc[["date_first_appeared","num_papers"]].copy() if "num_papers" in dc.columns else dc.copy()

    def append_if_missing(date_obj, count):
        ts = pd.Timestamp(date_obj)
        if (base["date_first_appeared"] == ts).any():
            return
        base.loc[len(base)] = {"date_first_appeared": ts, "num_papers": int(count)}

    if dow in (0,1,2,3,4):  # Mon..Fri runs (your schedule posts Sun–Thu at 17:00 UTC; Fri here only if you run manually)
        cnt = fetch_hepth_new_count_current()
        append_if_missing(d, cnt)
    elif dow == 6:  # Sunday: /new shows Friday; add Fri count + 0 for Sat/Sun
        fri, sat, sun = d - pd.Timedelta(days=2), d - pd.Timedelta(days=1), d
        cnt = fetch_hepth_new_count_current()  # Friday’s total on Sunday’s /new page
        append_if_missing(fri, cnt)
        append_if_missing(sat, 0)
        append_if_missing(sun, 0)

    # rebuild full features and overwrite daily_counts.csv
    full = build_features(daily_counts=base, holidays_df=hol, conference_df=conf)
    full = full.sort_values("date_first_appeared").reset_index(drop=True)
    full.to_csv(DAILY_CSV, index=False)
    return full, hol, conf  # return feature-augmented df

def predict_tomorrow(dc, hol, conf, model):
    # compute features for the next day from counts-only view to avoid any stale columns
    next_date = today_dub() + pd.Timedelta(days=1)
    base = dc[["date_first_appeared","num_papers"]].copy()
    X = build_features(daily_counts=base, holidays_df=hol, conference_df=conf, next_date=next_date)
    X = X.drop(columns=["date_first_appeared","num_papers"], errors="ignore")
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = 0
    X = X.reindex(columns=FEATURE_COLS)
    yhat = float(lgb_predict_any(model, X)[0])
    return next_date.date(), yhat

def compose_tweet(next_day, yhat, dc, log):
    # format helpers
    def fmt_d(d): return pd.Timestamp(d).strftime("%d/%m/%Y")

    last = dc.iloc[-1]
    last_date_ts = pd.to_datetime(last["date_first_appeared"])
    last_date = last_date_ts.date()
    last_val = int(last["num_papers"])
    yhat_round = int(round(yhat))

    # previous prediction error (rounded to int)
    err_txt = ""
    if "date" in log.columns and "yhat" in log.columns and len(log):
        log["date"] = pd.to_datetime(log["date"], errors="coerce")
        mask = log["date"].dt.date == last_date
        if mask.any():
            prev_pred = float(log.loc[mask, "yhat"].iloc[0])
            err_int = int(round(abs(prev_pred - last_val)))
            err_txt = f"\n• Error on {fmt_d(last_date)}: {err_int}"
            # write back error & actual
            log.loc[mask, ["err_abs","actual"]] = [err_int, last_val]
            log.to_csv(LOG_CSV, index=False)

    return (
        f"hep-th forecast for {fmt_d(next_day)}:\n"
        f"• Predicted new papers: {yhat_round}\n"
        f"• Last observed ({fmt_d(last_date)}): {last_val}"
        f"{err_txt}\n"
        "#hepth #arXiv"
    )


def tweet(status):
    # assumes: import os at module top
    if os.getenv("DRY_RUN", "0") == "1":
        print("[DRY_RUN] Would tweet:\n", status)
        return

    import tweepy  # ok to import here; DON'T import os here
    client = tweepy.Client(
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_SECRET"],
        wait_on_rate_limit=True,
    )
    try:
        resp = client.create_tweet(text=status)
        print("Tweet posted:", resp.data)
    except tweepy.Forbidden as e:
        # Helpful message if app lacks write or wrong product tier
        print("Twitter API Forbidden (check app permissions & tokens):", e)
        raise




def main():
    dc, hol, conf = update_daily_counts()
    model = load_any_lgb_model(MODEL_PATH)
    next_day, yhat = predict_tomorrow(dc, hol, conf, model)
    log = read_log()
    status = compose_tweet(next_day, yhat, dc, log)
    tweet(status)
    write_log(next_day, yhat)

if __name__ == "__main__":
    main()
