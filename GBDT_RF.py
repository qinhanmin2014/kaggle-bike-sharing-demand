import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

for col in ['casual', 'registered', 'count']:
    train['%s_log' % col] = np.log(train[col] + 1)
for df in [train, test]:
    date = pd.DatetimeIndex(df['datetime'])
    df['year'], df['month'], df['hour'], df['dayofweek'] = \
        date.year, date.month, date.hour, date.dayofweek
    df['hour_workingday_casual'] = df[['hour', 'workingday']].apply(
        lambda x: int(10 <= x['hour'] <= 19), axis=1)
    df['hour_workingday_registered'] = df[['hour', 'workingday']].apply(
      lambda x: int(
        (x['workingday'] == 1 and (x['hour'] == 8 or 17 <= x['hour'] <= 18))
        or (x['workingday'] == 0 and 10 <= x['hour'] <= 19)), axis=1)

preds = {}
for name, reg in [('gbdt', GradientBoostingRegressor(
                       n_estimators=1000, random_state=0)),
                  ('rf', RandomForestRegressor(
                       n_estimators=1000, random_state=0, n_jobs=-1))]:
    features = ['season', 'holiday', 'workingday', 'weather',
                'temp', 'atemp', 'humidity', 'windspeed',
                'year', 'hour', 'dayofweek', 'hour_workingday_casual']
    reg.fit(train[features], train['casual_log'])
    pred_casual = reg.predict(test[features])
    pred_casual = np.exp(pred_casual) - 1
    pred_casual[pred_casual < 0] = 0
    features = ['season', 'holiday', 'workingday', 'weather',
                'temp', 'atemp', 'humidity', 'windspeed',
                'year', 'hour', 'dayofweek', 'hour_workingday_registered']
    reg.fit(train[features], train['registered_log'])
    pred_registered = reg.predict(test[features])
    pred_registered = np.exp(pred_registered) - 1
    pred_registered[pred_registered < 0] = 0
    preds[name] = pred_casual + pred_registered

pred = 0.7 * preds['gbdt'] + 0.3 * preds['rf']
submission = pd.DataFrame({'datetime': test.datetime, 'count': pred},
                          columns=['datetime', 'count'])
submission.to_csv("submission/submission.csv", index=False)