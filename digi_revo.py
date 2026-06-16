import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title='The Indian Digital Revolution - Stock Analysis Dashboard',
    page_icon='📈',
    layout='wide'
)

st.title("India's Digital Revolution")
st.subheader("Performance of New-Age Indian Companies")
st.markdown('---')

stocks = {
    'Paytm 💳' : 'PAYTM.NS' ,
    'Zomato 🍕' : 'ETERNAL.NS',
    'Nykaa 💄' : 'NYKAA.NS',
    'Tata Motors EV 🚗' : 'TMPV.NS',
    'IRCTC 🚂' : 'IRCTC.NS'
}

st.sidebar.header('Settings🔨')
start_date = st.sidebar.date_input(
    "Start Date",
    value= pd.to_datetime('2022-01-01')
)
end_date = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime('2026-06-12')
)

if start_date >= end_date:
    st.error("❌ **Validation Error:** 'Start Date' cannot be greater than or equal to 'End Date'. Please select a valid timeline from the sidebar.")
    st.stop()

selected_stock_name = st.sidebar.selectbox('Select Stock' , list(stocks.keys()))
selected_ticker = stocks[selected_stock_name]


def load_data(start ,end):
    df_tata = yf.download('TMPV.NS' , start=start , end=end)
    df_paytm = yf.download('PAYTM.NS' , start=start, end=end)
    df_zom = yf.download('ETERNAL.NS', start= start , end=end)
    df_nykaa = yf.download('NYKAA.NS' , start=start , end=end)
    df_irctc = yf.download('IRCTC.NS', start=start, end=end)
    df_nifty = yf.download('^NSEI' , start=start , end = end)
    return df_tata , df_paytm , df_zom , df_nykaa , df_irctc , df_nifty

df_tata , df_paytm , df_zom , df_nykaa , df_irctc , df_nifty = load_data(start_date, end_date)

if selected_ticker == 'PAYTM.NS':
    df_selected = df_paytm
elif selected_ticker == 'ETERNAL.NS':
    df_selected = df_zom
elif selected_ticker == 'NYKAA.NS':
    df_selected = df_nykaa
elif selected_ticker == 'TMPV.NS':
    df_selected = df_tata
else:
    df_selected = df_irctc

#EDA

#A. Total Return -
start_price = df_selected['Close'].iloc[0]
curr_price = df_selected['Close'].iloc[-1]
total_return = ((curr_price - start_price)/start_price) * 100

#B. Volatility -
selected_dr = df_selected['Close'].pct_change()
volatility = selected_dr.std() * 100

#c. Normalised - 
selected_norm = (df_selected['Close'] / start_price) * 100
curr_norm_price = selected_norm.iloc[-1]

st.write(f"## Key Metric for {selected_stock_name}")
col1 , col2  = st.columns(2)

with col1:
    st.metric(
        label='Total Return',
        value=f"{round(float(total_return.iloc[0]) , 2)} %",
        delta=f"{round(float(total_return.iloc[0]) , 2)} %"
    )

with col2:
    st.metric(
        label='Volatility(Risk Measure)',
        value=f"{round(float(volatility.iloc[0]) , 2)} %"
    )

st.markdown('---')

st.write("Price Movement Chart")
fig1 = go.Figure()
if selected_stock_name == 'Tata Motors EV 🚗':
    df_selected = df_selected[df_selected.index >= '2025-10-14']

fig1.add_trace(go.Scatter(
    x = df_selected.index,
    y = df_selected['Close'].iloc[:,0],
    mode='lines',
    name=selected_stock_name
))

fig1.update_layout(
    title = f'{selected_stock_name} Live Price Trajectory',
    xaxis_title = 'Date',
    yaxis_title = 'Price (Rs)',
    template = 'plotly_dark'
)

st.plotly_chart(fig1, use_container_width=True)

st.markdown('---')
st.write("## Normalized Chart (Invested 100Rs)")

col3 , col4 = st.columns(2)

with col3:
    st.metric(
        label='Starting Investment',
        value='100 Rs'
    )

with col4:
    st.metric(
        label='Current Value',
        value=f"{round(float(curr_norm_price.iloc[0]) , 2)} Rs"
    )

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x = df_selected.index,
    y = selected_norm.iloc[:,0],
    mode='lines',
    name = selected_stock_name
))

fig2.update_layout(
    title = f'Normalized Chart for {selected_stock_name}',
    xaxis_title = 'Date',
    yaxis_title = 'Price(Normalized to 100 Rs)',
    template = 'plotly_dark'
)

st.plotly_chart(fig2 , use_container_width=True)

st.markdown('---')

st.write("## Comparing all stocks")
stocks_dict = {
    'Paytm 💳' : df_paytm ,
    'Zomato 🍕' : df_zom,
    'Nykaa 💄' : df_nykaa,
    'Tata Motors EV 🚗' : df_tata,
    'IRCTC 🚂' : df_irctc
}

fig3 = go.Figure()
for name,df in stocks_dict.items():
    clean_close = df['Close'].dropna()
    if name == 'Tata Motors EV 🚗':
        if end_date < pd.to_datetime('2025-10-14'):
            clean_close = pd.Series(dtype='float64')
        else:
            clean_close = clean_close[clean_close.index >= '2025-10-14']

    if len(clean_close) > 0:
        norm = (clean_close/clean_close.iloc[0]) * 100
        fig3.add_trace(go.Scatter(
            x = clean_close.index,
            y= norm.iloc[:,0],
            mode='lines',
            name = name
        ))
    else:
        st.warning(f"⚠️ No trading data available for {name} in this specific date range.({name} was listed on 14/10/2025)")

norm_nifty = (df_nifty['Close'] / df_nifty['Close'].iloc[0]) * 100
fig3.add_trace(go.Scatter(
    x = df_nifty.index,
    y = norm_nifty.iloc[:,0],
    mode='lines',
    line=dict(dash = 'dot' , width = 3 , color = 'White'),
    name='Nifty 50 (Market Benchmark)'
))

fig3.update_layout(
    title = 'ALL STOCK NORMALISED COMPARISION',
    xaxis_title = 'Date',
    yaxis_title = 'Price (Rs)',
    template = 'plotly_dark'
)

st.plotly_chart(fig3 , use_container_width=True)

st.markdown('---')

best_stocks = {}
for name , df in stocks_dict.items():
    clean_close = df['Close'].dropna()
    if name == 'Tata Motors EV 🚗':
        if end_date < pd.to_datetime('2025-10-14'):
            clean_close = pd.Series(dtype= 'float64')
        else:
            clean_close = clean_close[clean_close.index >= '2025-10-14']
    
    if len(clean_close) > 0:
        t_return = ((clean_close.iloc[-1] - clean_close.iloc[0])/clean_close.iloc[0]) * 100
        vol = clean_close.pct_change().std() * 100
        performance_score = float(t_return.iloc[0]) / float(vol.iloc[0])
        best_stocks[name] = performance_score
    else:
        st.warning(f"⚠️ No trading data available for {name} to calculate Leaderboard ranking.({name} was listed on 14/10/2025)")

sorted_stocks = sorted(best_stocks.items() , key=lambda x: x[1] , reverse=True)

winner_name = sorted_stocks[0][0]
winner_score = sorted_stocks[0][1]

loser_name = sorted_stocks[-1][0]
loser_score = sorted_stocks[-1][1]

st.info(f"📊 **Market Analysis Period:** Analyzing the performance of New-Age Indian Tech Giants from **{start_date.strftime('%B %Y')}** to **{end_date.strftime('%B %Y')}**.")

st.success(f"🥇 **{winner_name}** emerged as the absolute market leader during this timeframe. It delivered the highest efficiency with a Performance Score of **{round(winner_score, 2)}**, proving it generated the best returns relative to its daily volatility.")

st.warning(f"⚠️ On the flip side, **{loser_name}** heavily struggled under current market dynamics, recording the lowest risk-adjusted performance score of **{round(loser_score, 2)}** due to either low returns or excessive price fluctuations.")

st.write('## 🏆 Leaderboard -')
medals = ['🥇' , '🥈' , '🥉' , '4️⃣', '5️⃣']

for i , (name,score) in enumerate(sorted_stocks):
    col_a ,col_b , col_c = st.columns(3)
    with col_a:
        st.write(f'### {medals[i]}')
    
    with col_b:
        st.write(f'### {name}')
    
    with col_c:
        st.metric(
            label='Performance Score',
            value=f'{round(score,3)}'
        )

st.info('Performance Score = Total Return / Volatility')

st.markdown('---')

st.write(f'## 💰 Interactive SIP Calculator for {selected_stock_name}')

st.sidebar.markdown('---')
st.sidebar.header('SIP Calculator Settings 💵')
monthly_sip = st.sidebar.number_input('Enter Monthly SIP Amount (Rs)', value=1000 , step=500)

sip_df = df_selected['Close'].dropna()

if selected_stock_name == 'Tata Motors EV 🚗':
    sip_df = sip_df[sip_df.index >= '2025-10-14']

monthly_df = sip_df.resample('MS').first()

total_invested = 0
total_shares = 0

for date,row in monthly_df.iterrows():
    price = float(row.iloc[0])

    if price > 0 :
        total_invested += monthly_sip
        shares_bought = monthly_sip / price
        total_shares += shares_bought

latest_price = float(sip_df.iloc[-1].iloc[0])
current_value = total_shares * latest_price
net_profit_loss_rs = current_value - total_invested
abs_return_pct = (net_profit_loss_rs / total_invested) * 100 if total_invested > 0 else 0

col_sip1 , col_sip2 , col_sip3 = st.columns(3)

with col_sip1:
    st.metric(
        label='Total Invested Capital',
        value=f'Rs {total_invested:,}'
    )

with col_sip2:
    st.metric(
        label='Current Wealth Value',
        value=f"Rs {round(current_value , 2) :,}"
    )

with col_sip3:
    st.metric(
        label='Net Profit/Loss %',
        value=f"{round(abs_return_pct , 2)} %",
        delta=f'{round(abs_return_pct , 2)} %'
    )

st.info(f"💡 **SIP Insight:** This simulation assumes you invested **Rs {monthly_sip:,}** on the first trading day of every month, starting from the available data period.")


### Building the Model - 
#A. Collecting Data - 

def create_ml_features(sip_df):
    ml_df = pd.DataFrame(index=sip_df.index)
    ml_df['Close'] = sip_df

    ml_df['Momentum'] = ((ml_df['Close'] - ml_df['Close'].shift(3)) / ml_df['Close'].shift(3)) * 100
    ml_df['Volatility'] = ml_df['Close'].rolling(window=5).std()
    ml_df['Next_Close'] = ml_df['Close'].shift(-1)

    ml_df['Target'] = np.where(ml_df['Next_Close'] > ml_df['Close'] , 1 , 0)

    return ml_df

ml_df = create_ml_features(sip_df)
ml_df = ml_df.dropna()

#B. Data Preprocessing -

total_rows = len(ml_df)
split_index = int(total_rows * 0.8)

train_df = ml_df.iloc[:split_index]
test_df = ml_df.iloc[split_index:]

feature_cols = ['Momentum' , 'Volatility']
X_train = train_df[feature_cols]
y_train = train_df['Target']

X_test = test_df[feature_cols]
y_test = test_df['Target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled,y_train)

y_test_pred = model.predict(X_test_scaled)
model_accuracy = accuracy_score(y_test_pred,y_test)
accuracy_pct = model_accuracy * 100


st.markdown('---')
st.write('## 🤖 Trend Predictor ')
st.metric(
    label='Model Predictive Accuracy (Logistic Regression)',
    value=f"{accuracy_pct:.2f} %"
)
st.info("💡 **Note:** An accuracy above 50 % in stock markets is statistically significant due to high market noise and random walk dynamics.")

latest_day_row = ml_df.tail(1)
latest_feature = latest_day_row[['Momentum','Volatility']]

latest_feature_scaled = scaler.transform(latest_feature)

if st.button("🔮 Predict Tomorrow's Trend Direction"):
    probabilities = model.predict_proba(latest_feature_scaled)[0]
    prob_up = probabilities[1]*100
    prob_down = probabilities[0] * 100

    st.write("### 📊 Live Forecast Result:")

    if prob_up >= 50:
        st.success(f"🟢 **UPWARD TREND PREDICTED:** The model projects a **{round(prob_up, 2)}%** probability that {selected_stock_name}'s closing price will rise tomorrow.")
    else:
        st.error(f"🔴 **DOWNWARD TREND PREDICTED:** The model projects a **{round(prob_down, 2)}%** probability that {selected_stock_name}'s closing price will drop tomorrow.")

    st.caption(f"Prediction generated using the latest market closing data from index date: {latest_day_row.index[0].strftime('%Y-%m-%d')}")