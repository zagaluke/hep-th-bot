import os, re, requests, pandas as pd, numpy as np, joblib, tweepy\
from bs4 import BeautifulSoup\
from features import build_features\
\
DATA_DIR   = os.getenv("DATA_DIR", "data")\
DAILY_CSV  = os.path.join(DATA_DIR, "daily_counts.csv")\
HOL_CSV    = os.path.join(DATA_DIR, "holidays.csv")\
CONF_CSV   = os.path.join(DATA_DIR, "conferences.csv")\
MODEL_PATH = os.getenv("MODEL_PATH", "GBTmodel.pkl")\
LOG_CSV    = os.path.join(DATA_DIR, "pred_log.csv")\
TZ         = "Europe/Dublin"\
UA         = "hepth-bot/0.1 (mailto:your.email@example.com)"\
\
def today_dub(): return pd.Timestamp.now(tz=TZ).normalize()\
def dow_dub(): return int(today_dub().dayofweek)  # Mon=0..Sun=6\
def load_csvs():\
    dc  = pd.read_csv(DAILY_CSV, parse_dates=["date_first_appeared"]) if os.path.exists(DAILY_CSV) else pd.DataFrame(columns=["date_first_appeared","num_papers"])\
    hol = pd.read_csv(HOL_CSV, parse_dates=["date"])\
    conf= pd.read_csv(CONF_CSV, parse_dates=["start_date","end_date"])\
    if len(dc): dc = dc.sort_values("date_first_appeared").reset_index(drop=True)\
    return dc, hol, conf\
def save_dc(df): df.sort_values("date_first_appeared").to_csv(DAILY_CSV, index=False)\
def read_log():\
    if not os.path.exists(LOG_CSV): pd.DataFrame(columns=["date","yhat","tweeted_at","err_abs","actual"]).to_csv(LOG_CSV, index=False)\
    return pd.read_csv(LOG_CSV, parse_dates=["date","tweeted_at"])\
def write_log(date, yhat): \
    log = read_log()\
    log = pd.concat([log, pd.DataFrame([\{"date": pd.Timestamp(date), "yhat": float(yhat), "tweeted_at": pd.Timestamp.now(tz=TZ)\}])], ignore_index=True)\
    log.to_csv(LOG_CSV, index=False)\
\
def fetch_hepth_new_count_current():\
    r = requests.get("https://arxiv.org/list/hep-th/new?show=2000", headers=\{"User-Agent": UA\}, timeout=30)\
    r.raise_for_status()\
    soup = BeautifulSoup(r.text, "html.parser")\
    h3 = soup.find("h3", string=re.compile(r"^New submissions", re.I))\
    if not h3: return 0\
    m = re.search(r"showing\\s+(\\d+)\\s+of\\s+(\\d+)\\s+entries", h3.get_text(), re.I)\
    if m: return int(m.group(2))\
    cnt = 0\
    for el in h3.find_all_next():\
        if el.name == "h3": break\
        if el.name == "dt": cnt += 1\
    return cnt\
\
def append_if_missing(dc, date_obj, count):\
    ts = pd.Timestamp(date_obj)\
    if (dc["date_first_appeared"] == ts).any(): return dc\
    return pd.concat([dc, pd.DataFrame([\{"date_first_appeared": ts, "num_papers": int(count)\}])], ignore_index=True)\
\
def update_daily_counts():\
    dc, hol, conf = load_csvs()\
    d = today_dub().date()\
    dow = dow_dub()\
    if dow in (0,1,2,3,4):  # Mon..Thu (and Mon if you run then)\
        cnt = fetch_hepth_new_count_current()\
        dc = append_if_missing(dc, d, cnt)\
    elif dow == 6:  # Sunday: /new shows Friday; also add 0 for Sat and Sun\
        fri = d - pd.Timedelta(days=2)\
        sat = d - pd.Timedelta(days=1)\
        sun = d\
        cnt = fetch_hepth_new_count_current()  # Friday\'92s total exposed on Sunday\
        dc = append_if_missing(dc, fri, cnt)\
        dc = append_if_missing(dc, sat, 0)\
        dc = append_if_missing(dc, sun, 0)\
    save_dc(dc)\
    return dc, hol, conf\
\
def predict_tomorrow(dc, hol, conf, model):\
    next_date = today_dub() + pd.Timedelta(days=1)\
    X = build_features(daily_counts=dc, holidays_df=hol, conference_df=conf, next_date=next_date)\
    cols = [c for c in X.columns if c != "date_first_appeared"]\
    return next_date.date(), float(model.predict(X[cols])[0])\
\
def compose_tweet(next_day, yhat, dc, log):\
    last = dc.iloc[-1]\
    last_date = pd.to_datetime(last["date_first_appeared"]).date()\
    last_val = int(last["num_papers"])\
    trailing_7 = dc["num_papers"].tail(7).mean()\
    err_txt = ""\
    prev = log.loc[log["date"].dt.date == last_date, "yhat"]\
    if len(prev):\
        err = abs(float(prev.iloc[0]) - last_val)\
        err_txt = f"\\n\'95 Error on \{last_date\}: \{err:.1f\}"\
        log.loc[log["date"].dt.date == last_date, ["err_abs","actual"]] = [err, last_val]\
        log.to_csv(LOG_CSV, index=False)\
    return (\
        f"hep-th forecast for \{next_day\}:\\n"\
        f"\'95 Predicted new papers: \{int(round(yhat))\}\\n"\
        f"\'95 Last observed (\{last_date\}): \{last_val\}\\n"\
        f"\'95 7-day average: \{trailing_7:.1f\}"\
        f"\{err_txt\}\\n"\
        "#hepth #arXiv"\
    )\
\
def tweet(status):\
    auth = tweepy.OAuth1UserHandler(os.environ["X_API_KEY"], os.environ["X_API_SECRET"], os.environ["X_ACCESS_TOKEN"], os.environ["X_ACCESS_SECRET"])\
    tweepy.API(auth).update_status(status=status)\
\
def main():\
    dc, hol, conf = update_daily_counts()\
    model = joblib.load(MODEL_PATH)\
    next_day, yhat = predict_tomorrow(dc, hol, conf, model)\
    log = read_log()\
    status = compose_tweet(next_day, yhat, dc, log)\
    tweet(status)\
    write_log(next_day, yhat)\
\
if __name__ == "__main__":\
    main()\
}