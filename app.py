import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import xgboost as xgb
import numpy as np

st.set_page_config(page_title="Insurance Pricing Simulator", layout="wide")

st.title("Insurance Pricing Simulator")

@st.cache_data
def load_data():
    conn = sqlite3.connect('insurance_pricing.db')
    query = '''
    SELECT p.policy_id, p.age, p.vehicle_type, p.location, p.exposure,
           c.claim_count, c.claim_amount
    FROM policyholders p
    JOIN claims c ON p.policy_id = c.policy_id
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

df = load_data()
df['age_bucket'] = pd.cut(
    df['age'],
    bins=[17, 25, 40, 60, 100],
    labels=['18-25', '26-40', '41-60', '61+']
)

st.sidebar.header("Filter")
vehicle_filter = st.sidebar.multiselect("Vehicle Type", df['vehicle_type'].unique(), default=list(df['vehicle_type'].unique()))
location_filter = st.sidebar.multiselect("Location", df['location'].unique(), default=list(df['location'].unique()))

filtered_df = df[df['vehicle_type'].isin(vehicle_filter) & df['location'].isin(location_filter)]

st.subheader("Claim Frequency by Age")
freq_by_age = filtered_df.groupby('age_bucket').agg({
    'claim_count': 'sum',
    'exposure': 'sum'
})
freq_by_age['frequency'] = freq_by_age['claim_count'] / freq_by_age['exposure']

fig1, ax1 = plt.subplots()
freq_by_age['frequency'].plot(kind='bar', ax=ax1)
ax1.set_ylabel("Claims per Policy-Year")
ax1.set_xlabel("Age Bucket")
ax1.set_title("Claim Frequency by Age Group")
st.pyplot(fig1)

st.subheader("Claim Severity Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df[filtered_df['claim_amount'] > 0]['claim_amount'], bins=50, ax=ax2)
ax2.set_xlabel("Total Claim Amount")
ax2.set_title("Distribution of Non-Zero Claim Amounts")
st.pyplot(fig2)

st.subheader("Predictive Modeling")

model_input = pd.get_dummies(filtered_df[['age_bucket', 'vehicle_type', 'location']], drop_first=True)
model_input = model_input.reindex(columns=[
    'age_bucket_26-40', 'age_bucket_41-60', 'age_bucket_61+',
    'vehicle_type_SUV', 'vehicle_type_Truck', 'vehicle_type_Van',
    'location_Suburban', 'location_Urban'
], fill_value=0)

model = xgb.XGBRegressor(objective='count:poisson', n_estimators=100, max_depth=4)
X = model_input
y = filtered_df['claim_count']
model.fit(X, y)

preds = model.predict(X)
filtered_df['predicted_claims'] = preds
filtered_df['expected_loss'] = filtered_df['predicted_claims'] * 2000  #expected severity per claim = 2 * 1000 = 2000

st.dataframe(filtered_df[['policy_id', 'age', 'vehicle_type', 'location', 'claim_count', 'predicted_claims', 'expected_loss']].head(20))

st.subheader("Average Expected Loss by Age Group")
loss_by_age = filtered_df.groupby('age_bucket')['expected_loss'].mean()

fig3, ax3 = plt.subplots()
loss_by_age.plot(kind='bar', ax=ax3, color='tomato')
ax3.set_ylabel("Expected Loss ($)")
ax3.set_xlabel("Age Bucket")
ax3.set_title("Avg. Expected Loss per Policy")
st.pyplot(fig3)

st.download_button(
    "Download Predictions as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="predicted_claims.csv",
    mime="text/csv"
)

