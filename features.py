def _to_dates(s): return pd.to_datetime(s).dt.tz_localize(None).dt.normalize()\
\
def build_features(daily_counts, holidays_df, conference_df, new_counts_df=None, next_date=None):\
    base = daily_counts[['date_first_appeared','num_papers']].copy()\
    base['date_first_appeared'] = _to_dates(base['date_first_appeared'])\
    if new_counts_df is not None:\
        future = new_counts_df[['date_first_appeared','num_papers']].copy()\
        future['date_first_appeared'] = _to_dates(future['date_first_appeared'])\
        full = pd.concat([base,future],ignore_index=True).sort_values('date_first_appeared')\
    elif next_date is not None:\
        full = pd.concat([base,pd.DataFrame(\{'date_first_appeared':[_to_dates(pd.to_datetime(next_date))],'num_papers':[np.nan]\})],ignore_index=True)\
    else:\
        full = base.copy()\
    start_date = base['date_first_appeared'].min()\
    full['day'] = full['date_first_appeared'].dt.day\
    full['weekday'] = full['date_first_appeared'].dt.weekday\
    full['days_since_start'] = (full['date_first_appeared']-start_date).dt.days\
\
    # holidays\
    hol = _to_dates(holidays_df['date']).sort_values().to_numpy(dtype='datetime64[D]')\
    dates = full['date_first_appeared'].to_numpy(dtype='datetime64[D]')\
    def ds_last(d,ev):\
        i=np.searchsorted(ev,d,'right')-1\
        return np.nan if i<0 else (d-ev[i]).astype('timedelta64[D]').astype(int)\
    def du_next(d,ev):\
        i=np.searchsorted(ev,d,'left')\
        return np.nan if i>=len(ev) else (ev[i]-d).astype('timedelta64[D]').astype(int)\
    full['days_since_last_public_holiday']=[ds_last(d,hol) for d in dates]\
    full['days_until_public_holiday']=[du_next(d,hol) for d in dates]\
    is_hol=np.isin(dates,hol)\
    full.loc[is_hol,'days_since_last_public_holiday']=0; full.loc[is_hol,'days_until_public_holiday']=0\
\
    # conferences (interval-based)\
    conf = conference_df.copy()\
    conf['start_date'] = _to_dates(conf['start_date'])\
    conf['end_date']   = _to_dates(conf['end_date'])\
    starts, ends = conf['start_date'].to_numpy(), conf['end_date'].to_numpy()\
    days_since, days_until = [], []\
    for d in full['date_first_appeared']:\
        in_any = ((starts <= d) & (ends >= d))\
        if in_any.any():\
            days_since.append(0); days_until.append(0); continue\
        past_ends = ends[ends < d]\
        future_starts = starts[starts > d]\
        days_since.append((d - past_ends.max()).days if len(past_ends) else np.nan)\
        days_until.append((future_starts.min() - d).days if len(future_starts) else np.nan)\
    full['days_since_last_conference']=days_since\
    full['days_until_next_conference']=days_until\
\
    # lags & rolls\
    full=full.sort_values('date_first_appeared')\
    full['lag_1']=full['num_papers'].shift(1); full['lag_7']=full['num_papers'].shift(7)\
    for w in [7,30,90,365]: full[f'roll_\{w\}']=full['num_papers'].rolling(w,min_periods=1).mean()\
    full['mean_same_weekday_4w']=np.nan\
    for wd in range(7):\
        m=full['weekday']==wd\
        full.loc[m,'mean_same_weekday_4w']=full.loc[m,'num_papers'].rolling(4,min_periods=1).mean().values\
\
    cols=['date_first_appeared','num_papers','days_until_public_holiday','days_since_last_public_holiday','day','weekday','days_since_start','mean_same_weekday_4w','days_since_last_conference','roll_7','roll_30','roll_90','roll_365','lag_1','lag_7','days_until_next_conference']\
    if new_counts_df is not None: return full.loc[full['date_first_appeared'].isin(_to_dates(new_counts_df['date_first_appeared'])),cols].reset_index(drop=True)\
    if next_date is not None: return full.loc[full['date_first_appeared']==_to_dates(pd.to_datetime(next_date)),cols].reset_index(drop=True)\
    return full[cols]\
}