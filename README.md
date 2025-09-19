# hep-th daily forecast bot (arXiv)

Predicts tomorrow’s hep-th “new” paper count and tweets it nightly.
Runs via GitHub Actions; maintains CSV history for features and evaluation.

# workflow

Scrapes daily hep-th “new” counts and maintains data/daily_counts.csv.

Builds features (lags/rolls/holidays/conferences) and predicts with LightGBM.

Logs predictions in data/pred_log.csv (target date = next business day).

Tweets status on Sun–Thu evenings; Sunday uses Friday as “last observed”.

Commits updated CSVs back to the repo.
