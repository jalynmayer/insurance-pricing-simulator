
import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n = 10_000

ages = np.random.randint(18, 80, size=n)
vehicle_type = np.random.choice(['Sedan', 'SUV', 'Truck', 'Van'],
                                size=n, p=[0.4, 0.3, 0.2, 0.1])
location = np.random.choice(['Urban', 'Suburban', 'Rural'],
                            size=n, p=[0.5, 0.3, 0.2])
exposure = np.ones(n)  

lambda = (0.02 +
       (ages - 18) / 1000 +
       (vehicle_type == 'Truck') * 0.01 +
       (location == 'Urban') * 0.005)

claim_count = np.random.poisson(lambda * exposure)


claim_amount = [
    np.random.gamma(shape=2, scale=1000, size=k).sum() if k > 0 else 0.0
    for k in claim_count
]

df_policy = pd.DataFrame({
    'policy_id': np.arange(1, n + 1),
    'age': ages,
    'vehicle_type': vehicle_type,
    'location': location,
    'exposure': exposure
})

df_claims = pd.DataFrame({
    'claim_id': np.arange(1, n + 1),
    'policy_id': np.arange(1, n + 1),
    'claim_count': claim_count,
    'claim_amount': claim_amount
})

conn = sqlite3.connect('insurance_pricing.db')
df_policy.to_sql('policyholders', conn, if_exists='replace', index=False)
df_claims.to_sql('claims', conn, if_exists='replace', index=False)

query = """
SELECT
    p.policy_id,
    p.age,
    p.vehicle_type,
    p.location,
    p.exposure,
    c.claim_count,
    c.claim_amount
FROM policyholders p
JOIN claims c ON p.policy_id = c.policy_id
"""
data = pd.read_sql_query(query, conn)
conn.close()


data['age_bucket'] = pd.cut(
    data['age'],
    bins=[17, 25, 40, 60, 100],
    labels=['18-25', '26-40', '41-60', '61+']
)

cat_cols = ['age_bucket', 'vehicle_type', 'location']
X_cat = pd.get_dummies(data[cat_cols], drop_first=True)

X = X_cat.copy()
y = data['claim_count']
exposure_clean = data['exposure']


X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = X.notnull().all(axis=1) & y.notnull()

X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)
exposure_clean = exposure_clean.loc[mask].reset_index(drop=True)


X_glm = sm.add_constant(X, has_constant='add')


X_glm = X_glm.apply(pd.to_numeric, errors='coerce').astype(float)
y       = y.astype(float)
offset  = np.log(exposure_clean).astype(float)

valid = X_glm.notnull().all(axis=1) & ~y.isna() & ~offset.isna()
X_glm = X_glm.loc[valid].to_numpy()           # convert to NumPy
y      = y.loc[valid].to_numpy()
offset = offset.loc[valid].to_numpy()

glm_poisson = sm.GLM(
    y,
    X_glm,
    family=sm.families.Poisson(),
    offset=offset
)
glm_result = glm_poisson.fit()
print("\n=============  Poisson GLM Summary  =============")
print(glm_result.summary())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


model = xgb.XGBRegressor(
    objective='count:poisson',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("\nXGBoost RMSE:", rmse)


manual_freq = (
    data.loc[mask]
        .groupby('age_bucket')['claim_count']
        .sum()
        / data.loc[mask].groupby('age_bucket')['exposure'].sum()
)
print("\nManual claim frequency by age bucket:\n", manual_freq)

print("\nSimulation complete!  SQLite DB saved as 'insurance_pricing.db'.")
