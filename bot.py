# bot.py


import sys
print(">>> Python executable:", sys.executable)
print(">>> sys.path[0]:", sys.path[0])

#import relevant packages

import os, re, requests, pandas as pd, numpy as np, joblib
from dotenv import load_dotenv; load_dotenv()
from bs4 import BeautifulSoup
from features import build_features
import lightgbm as lgb

#open files

DATA_DIR   = os.getenv("DATA_DIR", "data")
DAILY_CSV  = os.path.join(DATA_DIR, "daily_counts.csv")
HOL_CSV    = os.path.join(DATA_DIR, "holidays.csv")
CONF_CSV   = os.path.join(DATA_DIR, "conferences.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "GBTmodel.pkl")
LOG_CSV    = os.path.join(DATA_DIR, "pred_log.csv")
TZ         = "Europe/Dublin"
UA         = "hepth-bot/0.1 (mailto:your.email@example.com)"


def load_any_lgb_model(path):

    if path.lower().endswith(".txt"):
        return lgb.Booster(model_file=path)

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


#features

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

def today_dub():
    s = os.getenv("FAKE_TODAY") #for testing
    if s:
        return pd.Timestamp(s).tz_localize("Europe/Dublin").normalize()
    return pd.Timestamp.now(tz=TZ).normalize()

def dow_dub():
    return int(today_dub().dayofweek)  # Mon=0..Sun=6
    
def load_csvs():
    dc  = pd.read_csv(DAILY_CSV, parse_dates=["date_first_appeared"]) if os.path.exists(DAILY_CSV) else pd.DataFrame(columns=["date_first_appeared","num_papers"])
    hol = pd.read_csv(HOL_CSV, parse_dates=["date"]) if os.path.exists(HOL_CSV) else pd.DataFrame(columns=["date"])
    conf= pd.read_csv(CONF_CSV, parse_dates=["start_date","end_date"]) if os.path.exists(CONF_CSV) else pd.DataFrame(columns=["start_date","end_date"])
    if len(dc):
        dc = dc.sort_values("date_first_appeared").reset_index(drop=True)
    return dc, hol, conf
    
def save_dc(df):
    df = df.sort_values("date_first_appeared")
    _atomic_to_csv(df, DAILY_CSV)

def _atomic_to_csv(df: pd.DataFrame, path: str):
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def last_business_day_row(dc):
    m = dc["date_first_appeared"].dt.weekday <= 4   # Mon(0) .. Fri(4)
    if m.any():
        return dc.loc[m].iloc[-1]
    return dc.iloc[-1]  # fallback (shouldn't happen with your schedule)

def last_business_day(dc):
    return pd.to_datetime(last_business_day_row(dc)["date_first_appeared"]).normalize()


def read_log():
    # Ensure file exists with correct schema
    cols = ["date", "yhat", "tweeted_at", "err_abs", "actual"]
    if not os.path.exists(LOG_CSV):
        empty = pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns]"),
            "yhat": pd.Series([], dtype="float"),
            "tweeted_at": pd.Series([], dtype="datetime64[ns]"),
            "err_abs": pd.Series([], dtype="float"),
            "actual": pd.Series([], dtype="float"),
        })
        _atomic_to_csv(empty, LOG_CSV)

    # Tolerant read in case of malformed lines
    try:
        log = pd.read_csv(LOG_CSV, parse_dates=["date", "tweeted_at"], on_bad_lines="skip")
    except Exception:
        # Last resort: read without parse_dates then coerce; keep a backup
        try:
            os.replace(LOG_CSV, LOG_CSV + ".corrupt_backup")
        except Exception:
            pass
        log = pd.DataFrame(columns=cols)

    # Enforce schema and coerce types
    for c in cols:
        if c not in log.columns:
            log[c] = pd.NA
    log["date"] = pd.to_datetime(log["date"], errors="coerce").dt.tz_localize(None)
    log["tweeted_at"] = pd.to_datetime(log["tweeted_at"], errors="coerce")
    for c in ["yhat", "err_abs", "actual"]:
        log[c] = pd.to_numeric(log[c], errors="coerce")

    # De-dup by date (normalized), keep latest by tweeted_at
    if not log.empty:
        log["date_norm"] = log["date"].dt.normalize()
        log = (log.sort_values(["date_norm", "tweeted_at"], na_position="last")
                  .drop_duplicates(subset=["date_norm"], keep="last")
                  .drop(columns=["date_norm"]))
        log = log.sort_values("date")

    return log.reset_index(drop=True)
    
    
    
    
def write_prediction(pred_date, yhat):
    log = read_log()
    pred_date = pd.Timestamp(pred_date).normalize()
    row = {
        "date": pred_date,
        "yhat": float(yhat),
        "tweeted_at": pd.NaT,
        "err_abs": pd.NA,
        "actual": pd.NA,
    }
    if not log.empty and (log["date"].dt.normalize() == pred_date).any():
        m = log["date"].dt.normalize() == pred_date
        for k, v in row.items():
            if k == "tweeted_at":
                # do not overwrite tweeted_at here
                continue
            log.loc[m, k] = v
    else:
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)

    # De-dup (stay consistent with read_log)
    log["date_norm"] = log["date"].dt.normalize()
    log = (log.sort_values(["date_norm", "tweeted_at"], na_position="last")
              .drop_duplicates(subset=["date_norm"], keep="last")
              .drop(columns=["date_norm"]))
    log = log.sort_values("date").reset_index(drop=True)
    _atomic_to_csv(log, LOG_CSV)
    return log    


    
def mark_tweeted(pred_date):
    log = read_log()
    if log.empty:
        return log
    pred_date = pd.Timestamp(pred_date).normalize()
    m = log["date"].dt.normalize() == pred_date
    now = pd.Timestamp.now(tz=TZ)
    if m.any():
        # only set if missing to keep idempotency
        log.loc[m & log["tweeted_at"].isna(), "tweeted_at"] = now
        _atomic_to_csv(log, LOG_CSV)
    return log    

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

def update_yesterday_error(dc):
    log = read_log()
    if log.empty or "date" not in log.columns:
        return log

    obs_date = last_business_day(dc)                     # e.g., Friday on Sunday runs
    obs_val  = int(last_business_day_row(dc)["num_papers"])

    mask = log["date"].dt.normalize() == obs_date
    if mask.any():
        prev_pred = float(log.loc[mask, "yhat"].iloc[-1])
        err_int   = int(round(abs(prev_pred - obs_val)))
        log.loc[mask, ["err_abs", "actual"]] = [err_int, obs_val]
        _atomic_to_csv(log, LOG_CSV)
    return log



def compose_tweet(next_day, yhat, dc, log):
    def fmt_d(d): return pd.Timestamp(d).strftime("%d/%m/%Y")

    obs_row  = last_business_day_row(dc)
    obs_date = pd.to_datetime(obs_row["date_first_appeared"]).normalize()
    obs_val  = int(obs_row["num_papers"])
    yhat_round = int(round(yhat))

    err_txt = ""
    if not log.empty and "date" in log.columns and "err_abs" in log.columns:
        m = log["date"].dt.normalize() == obs_date
        if m.any() and not pd.isna(log.loc[m, "err_abs"].iloc[-1]):
            err_txt = f"\n• Error on {fmt_d(obs_date)}: {int(round(float(log.loc[m, 'err_abs'].iloc[-1])))}"

    return (
        f"hep-th forecast for {fmt_d(next_day)}:\n"
        f"• Predicted new papers: {yhat_round}\n"
        f"• Last observed ({fmt_d(obs_date)}): {obs_val}"
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
    # 0) refresh counts/features
    dc, hol, conf = update_daily_counts()
    model = load_any_lgb_model(MODEL_PATH)
    next_day, yhat = predict_tomorrow(dc, hol, conf, model)

    # 1) update error for last business day
    log = update_yesterday_error(dc)

    # 2) WRITE TODAY’S PREDICTION FIRST (so we never lose it)
    log = write_prediction(next_day, yhat)

    # 3) compose status using the (now up-to-date) log
    status = compose_tweet(next_day, yhat, dc, log)

    # 4) tweet (or DRY_RUN prints), then mark_tweeted on success
    try:
        tweet(status)
        mark_tweeted(next_day)
    except Exception as e:
        # Keep the prediction row even if tweeting fails
        print("Tweeting failed; prediction was logged. Error:", repr(e))
        # Do not re-raise, so the workflow can still commit updated CSVs

if __name__ == "__main__":
    main()
