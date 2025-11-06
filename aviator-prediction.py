"""
Aviator Predictor - Streamlit App
File: aviator_predictor_app.py

How to run:
1. Create a virtualenv and activate it.
2. Install dependencies:
   pip install streamlit pandas numpy matplotlib statsmodels scikit-learn
   (statsmodels and scikit-learn are optional; the app will work without them but AR prediction will be disabled.)
3. Run:
   streamlit run aviator_predictor_app.py

Purpose:
- Accept manual comma-separated multipliers or CSV upload
- Compute statistics (mean, median, std, min, max)
- Compute rolling means, EWMA
- Provide simple AR(3) forecast if statsmodels installed
- Show bucket transition probabilities
- Display plots and a recommended next-range prediction

Note: This app is for learning and analysis. Do not use for real betting decisions.

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import math

# Try to import statsmodels for AR; if not available, we fallback
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATSmodels_AVAILABLE = True
except Exception:
    STATSmodels_AVAILABLE = False

st.set_page_config(page_title="Aviator Predictor", layout="wide")

# ---------------------- Utility functions ----------------------

def parse_input(text_or_file):
    """Parse comma-separated text or a CSV file-like object into list of floats."""
    if text_or_file is None:
        return []
    if hasattr(text_or_file, 'read'):
        # file-like (Uploaded file)
        content = text_or_file.read().decode('utf-8')
    else:
        content = str(text_or_file)
    # accept commas, spaces, newlines
    for ch in ['\n', '\r', ';']:
        content = content.replace(ch, ',')
    parts = [p.strip() for p in content.split(',') if p.strip() != '']
    values = []
    for p in parts:
        try:
            # allow trailing x like '3.3x'
            p_clean = p.lower().replace('x', '')
            v = float(p_clean)
            values.append(v)
        except Exception:
            # ignore non-numeric tokens
            continue
    return values


def compute_basic_stats(values):
    arr = np.array(values, dtype=float)
    return {
        'count': len(arr),
        'mean': float(np.mean(arr)) if len(arr)>0 else None,
        'median': float(np.median(arr)) if len(arr)>0 else None,
        'std': float(np.std(arr, ddof=1)) if len(arr)>1 else 0.0,
        'min': float(np.min(arr)) if len(arr)>0 else None,
        'max': float(np.max(arr)) if len(arr)>0 else None
    }


def add_features(df):
    # rolling stats
    for k in (3,5,10,30):
        df[f'roll_mean_{k}'] = df['value'].rolling(k, min_periods=1).mean()
        df[f'roll_std_{k}']  = df['value'].rolling(k, min_periods=1).std().fillna(0)
    df['ewma_03'] = df['value'].ewm(alpha=0.3).mean()
    df['ewma_05'] = df['value'].ewm(alpha=0.5).mean()
    # buckets
    bins = [-np.inf, 1.2, 2.0, 5.0, 10.0, np.inf]
    labels = ['<1.2','1.2-2','2-5','5-10','>10']
    df['bucket'] = pd.cut(df['value'], bins=bins, labels=labels)
    # consecutive below threshold
    df['below_1_2'] = (df['value'] < 1.2).astype(int)
    df['consec_below_1_2'] = df['below_1_2'] * (df['below_1_2'].groupby((df['below_1_2'] != df['below_1_2'].shift()).cumsum()).cumcount() + 1)
    return df


def ewma_predict(values, alpha=0.3):
    if len(values) == 0:
        return None
    return float(pd.Series(values).ewm(alpha=alpha).mean().iloc[-1])


def ar_predict(values, lags=3):
    if not STATSmodels_AVAILABLE:
        return None, 'statsmodels not installed'
    if len(values) < lags + 2:
        return None, 'not enough data for AR'
    try:
        logv = np.log1p(values)
        model = AutoReg(logv, lags=lags, old_names=False).fit()
        pred_log = model.predict(start=len(logv), end=len(logv))
        pred = float(np.expm1(pred_log[0]))
        return pred, None
    except Exception as e:
        return None, str(e)


def bucket_transition_prob(values):
    if len(values) < 2:
        return pd.DataFrame()
    df = pd.DataFrame({'value': values})
    bins = [-np.inf, 1.2, 2.0, 5.0, 10.0, np.inf]
    labels = ['<1.2','1.2-2','2-5','5-10','>10']
    s = pd.cut(df['value'], bins=bins, labels=labels)
    trans = pd.crosstab(s.shift(0), s.shift(-1))
    prob = trans.div(trans.sum(axis=1).replace(0,1), axis=0)
    return prob.fillna(0)


def recommend_range(values, ewma_val, std_val):
    # Simple heuristic: predicted range = EWMA +/- k*std (clipped to >= 1.0)
    if ewma_val is None:
        return (None, None)
    k = 0.9
    low = max(1.0, ewma_val - k * std_val)
    high = ewma_val + k * std_val
    return (round(low,2), round(high,2))

# ---------------------- Streamlit UI ----------------------

st.title("Aviator Pattern Analyzer & Next-Value Predictor")
st.markdown("Enter recent multipliers manually (comma-separated) or upload a CSV with one column of multipliers.")

col1, col2 = st.columns([2,1])
with col1:
    input_text = st.text_area("Paste comma-separated multipliers (e.g. 3.3,1.0,2.2,...)", height=120)
    uploaded_file = st.file_uploader("Or upload CSV file (one column with values)", type=['csv','txt'])
    use_sample = st.checkbox("Use sample data (demo)", value=False)
with col2:
    st.write("Settings")
    alpha = st.slider('EWMA alpha', min_value=0.05, max_value=0.9, value=0.3, step=0.05)
    show_ar = st.checkbox('Attempt AR(3) forecast (requires statsmodels)', value=STATSmodels_AVAILABLE)
    show_chart = st.checkbox('Show chart', value=True)

if use_sample:
    sample = "3.3,1,2.2,2.4,4,6.6,7.8,6.6,2"
    values = parse_input(sample)
elif uploaded_file is not None:
    values = parse_input(uploaded_file)
else:
    values = parse_input(input_text)

if len(values) == 0:
    st.info('No numeric multipliers provided yet. Paste values or upload a CSV, or check "Use sample data".')
    st.stop()

# create dataframe
df = pd.DataFrame({'value': values})
df = add_features(df)

# Basic stats
stats = compute_basic_stats(values)

# Predictions
ewma_val = ewma_predict(values, alpha=alpha)
ar_val, ar_err = ar_predict(values, lags=3) if show_ar else (None, 'AR disabled')

# Bucket transitions
prob_df = bucket_transition_prob(values)

# Recommendation
pred_low, pred_high = recommend_range(values, ewma_val, stats['std'] or 0.0)

# ---------------------- Output ----------------------

st.subheader('Summary Statistics')
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric('Count', stats['count'])
col_b.metric('Mean', round(stats['mean'],2))
col_c.metric('Median', round(stats['median'],2))
col_d.metric('Std Dev', round(stats['std'],2))

st.write('Min:', stats['min'], ' Max:', stats['max'])

st.subheader('Predictions')
cols1, cols2 = st.columns(2)
with cols1:
    st.write('EWMA (alpha={}):'.format(alpha), round(ewma_val,2))
    st.write('Recommended next-range (heuristic):', f"{pred_low}x  —  {pred_high}x")
with cols2:
    if ar_val is not None:
        st.success(f'AR(3) forecast: {round(ar_val,2)}x')
    else:
        st.info('AR forecast not available: ' + (ar_err or 'unknown'))

st.markdown('**Logic hint:**')
if values[-1] < stats['mean']:
    st.write('Last round below mean — possible short-term rebound (mean-reversion)')
else:
    st.write('Last round above mean — watch for pullback or continuation depending on momentum')

# show bucket transitions
st.subheader('Bucket Transition Probabilities (empirical)')
if not prob_df.empty:
    st.dataframe(prob_df)
else:
    st.write('Not enough data to compute transitions')

# show dataframe and last rows
st.subheader('Data (most recent last)')
st.dataframe(df.tail(200))

# chart
if show_chart:
    st.subheader('Chart: multipliers and EWMA')
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df.index+1, df['value'], marker='o', label='Value')
    ax.plot(df.index+1, df['ewma_03'], linestyle='--', label='EWMA(0.3)')
    # predicted next point
    next_x = df.index[-1]+2
    ax.scatter(next_x, ewma_val, marker='X', s=80, label='Predicted Next (EWMA)')
    ax.set_xlabel('Round #')
    ax.set_ylabel('Multiplier (x)')
    ax.set_ylim(bottom=0)
    ax.legend()
    st.pyplot(fig)

st.markdown('---')

st.subheader('Backtest suggestion (local)')
st.write('To evaluate performance, load a long history (e.g. 1000 rounds) and perform a rolling-window backtest where the predictor is computed only from past data and compared to actual next round values.')

st.markdown('---')
st.write('App created for learning and analysis. Remember: predictions are probabilistic, not guarantees.')

# Footer: quick download of processed data
csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download processed CSV', data=csv, file_name='aviator_processed.csv', mime='text/csv')
