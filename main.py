import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import ewstools
import os, io, re, glob, base64
import zipfile
from PIL import Image
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['figure.dpi'] = 200

st.set_page_config(page_title="Characterization of hibernators body temperature", layout="centered")

st.title("Characterization of the body temperature dynamics of hibernators")
#st.caption("Uploads a body temperature CSV and computes Variance, lag-1 Autocorrelation, and Kendall’s τ over a pre-hibernation window.")
st.sidebar.header("Inputs")

if(os.path.isfile('demo.zip')):
    os.remove('demo.zip')
with zipfile.ZipFile('demo.zip', 'x') as csv_zip:
    csv_zip.writestr("demo.csv",
                    pd.read_csv("demo.csv").to_csv(index=False))    
with open("demo.zip", "rb") as file:
    #st.sidebar.download_button(label = "Download demo data",data = file,file_name = "demo.zip")
    zip_data = file.read()
    b64 = base64.b64encode(zip_data).decode()
    zip_filename = 'demo.zip'
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">Download demo data</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
if(os.path.isfile('demo.zip')):
    os.remove('demo.zip')
    
uploaded = st.sidebar.file_uploader("CSV file (one column with body temperature)", type=["csv"])

column_name = st.sidebar.text_input(
    "Column name for body temperature (leave empty to use the first numeric column)",
    value=""
)

start_index = st.sidebar.number_input("start_index (row index to start)", min_value=0, value=464, step=1)
hib_index = st.sidebar.number_input("hibernation_index (first row of hibernation)", min_value=0, value=6877, step=1)

samples_per_hour = st.sidebar.number_input(
    "samples_per_hour (measurements per hour, e.g., 6 for every 10 min)",
    min_value=1, value=6, step=1
)

compute_btn = st.sidebar.button("Run computation")

st.markdown(
"""
- This app computes early warning signal (EWS) indicators on a **pre-hibernation** segment of hibernators body temperature.
- The analysis segment is `[start_index, hibernation_index)` (pre-hibernation only).
- `samples_per_hour` sets the time scale (e.g., 6 → every 10 min, 12 → every 5 min).
"""
)

def pick_temperature_column(df: pd.DataFrame, user_col: str) -> str:
    if user_col and user_col in df.columns:
        return user_col
    # Fallback: first numeric column
    numeric_cols = df.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found. Please specify the body temperature column name.")
    return numeric_cols[0]
   
if compute_btn:
    if uploaded is None:
        st.warning("Please upload a CSV file.")
        st.stop()

    # Read data
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    try:
        temp_col = pick_temperature_column(df, column_name)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Basic range checks
    n_rows = len(df)
    if start_index < 0 or hib_index <= start_index or hib_index > n_rows:
        st.error(
            f"Invalid indices: start_index={start_index}, hibernation_index={hib_index}, n_rows={n_rows}.\n"
            "Ensure 0 ≤ start_index < hibernation_index ≤ number of rows."
        )
        st.stop()

    series = df[temp_col].iloc[int(start_index):int(hib_index)].astype(float).to_numpy()

    if series.size < (24 * samples_per_hour * 5):
        st.warning(
            "The selected window is quite short (< ~5 days of data). "
            "EWS trends may be unreliable."
        )

    # --- Define hamster-specific analysis windows ---
    per_day = int(24 * samples_per_hour)   # samples per day
    w = int(20 * per_day)                  # rolling window = 20 days
    s_detrend = int(per_day)               # Lowess span = 1 day

    # Safety: require at least w+some buffer to compute rolling metrics
    if len(series) < max(w + 5, per_day * 2):
        st.warning(
            f"Selected segment has only {len(series)} samples while the 20-day window requires ~{w} samples.\n"
            "Variance/AC1 will be computed but may be truncated or empty if the segment is too short."
        )

    # --- EWS via ewstools ---
    try:
        ts = ewstools.TimeSeries(data=series)
        ts.detrend(method='Lowess', span=s_detrend)            # 1-day Lowess span
        ts.compute_var(rolling_window=w)                       # rolling variance
        ts.compute_auto(lag=1, rolling_window=w)               # lag-1 autocorrelation
        ts.compute_ktau()                                      # Kendall's tau for each indicator
    except Exception as e:
        st.error(f"EWS computation failed: {e}")
        st.stop()

    tau_df = pd.concat([pd.DataFrame(ts.ktau.values(), index=ts.ktau.keys())])
    tau_df = tau_df.rename(columns={0: 'Correlation'})  # consistent with your A table
    tau_df = tau_df.rename(index={'ac1': 'Autocorrelation', 'variance': 'Variance'})
    A = tau_df.loc[['Variance', 'Autocorrelation']].copy()

    st.subheader("Kendall’s τ (Correlation with time)")
    st.dataframe(A.style.format(precision=3), use_container_width=True)

    fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    try:
        full_temp = df[temp_col].astype(float).to_numpy()
        ax[0].plot(np.arange(0, len(full_temp)), full_temp, c='gray', alpha=0.7, linewidth=0.6)
    except Exception:
        ax[0].plot(np.arange(int(start_index), int(hib_index)), series, c='gray', alpha=0.7, linewidth=0.6)

    ax[0].axvline(int(start_index), color='black', linestyle='--', linewidth=1)
    ax[0].axvline(int(hib_index), color='black', linestyle='--', linewidth=1)
    ax[0].set_ylabel('T'+"$\mathrm{_B}$"+' (°C)')
    ax[0].yaxis.set_major_locator(MaxNLocator(6))

    # Panel 2: Rolling variance on the analysis segment's index range
    if not ts.ews.empty and 'variance' in ts.ews.columns:
        # ts.ews has an implicit time index starting at 0 for the segment
        x_var = np.arange(int(start_index), int(start_index) + len(ts.ews['variance']))
        ax[1].plot(x_var, ts.ews['variance'].to_numpy(), c="gray")
    ax[1].set_ylabel('Variance')
    ax[1].yaxis.set_major_locator(MaxNLocator(6))

    # Panel 3: Rolling lag-1 autocorrelation
    if not ts.ews.empty and 'ac1' in ts.ews.columns:
        x_ac = np.arange(int(start_index), int(start_index) + len(ts.ews['ac1']))
        ax[2].plot(x_ac, ts.ews['ac1'].to_numpy(), c="gray")
    ax[2].set_ylabel('Autocorrelation')
    ax[2].set_xlabel('Row index (time)')
    sns.despine()
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.markdown(
    """
    - Upload body temperature data in the following format.
    """
    )
    image = Image.open('demo.png')
    st.image(image, caption='')
        
with st.expander("What the app does (detailed explanation)"):
    st.markdown(
        """
**Overview**

This app computes early warning signal (EWS) indicators on a **pre-hibernation** segment of hibernators body temperature:
1. **Variance** (increases as hibernation approaches),
2. **Autocorrelation** (increases as hibernation approaches),
3. **Kendall’s τ (tau)** for each indicator vs. time, summarizing whether that indicator tends to **increase** (τ>0) or **decrease** (τ<0) as the transition approaches.

**Inputs**

- **CSV file**: provide a single column with body temperature values. If your file has multiple columns, specify the target column name in the sidebar.
- **start_index** and **hibernation_index**: determine the pre-hibernation window. The analysis uses rows `[start_index, hibernation_index)`, strictly excluding post-transition data.
- **samples_per_hour**: measurements per hour (e.g., `6` for 10-minute sampling). This sets the time scale required to build day-scale windows.

**Method**

- The series is first **detrended** with **Lowess** using a span of **1 day** (`span = 24 × samples_per_hour`).
- EWS are computed in **rolling windows** spanning **20 days** (`rolling_window = 20 × 24 × samples_per_hour`).
- We compute:
  - **Rolling Variance** over the detrended series,
  - **Rolling lag-1 Autocorrelation (Autocorrelation)** over the detrended series.
- Finally, **Kendall’s τ** is computed per indicator to summarize the **monotonic trend** in the pre-hibernation window. The table shows τ for **Variance** and **Autocorrelation**.

**Outputs**

- A **three-panel figure**:
  1. Full temperature trace with vertical dashed lines marking `start_index` and `hibernation_index`.
  2. Rolling **Variance** over the pre-hibernation segment.
  3. Rolling **Autocorrelation** over the pre-hibernation segment.
- A table with **Kendall’s τ** 

**Notes**

- If your pre-hibernation segment is **shorter than ~20 days**, the rolling calculations will be truncated or may produce very few data points. τ may then be unstable; consider supplying a longer pre-hibernation segment.
- Choose `samples_per_hour` carefully (e.g., `6` for every 10 minutes, `12` for every 5 minutes). This parameter controls window sizes.
- The app assumes clean, regularly sampled data without large gaps. If gaps exist, consider preprocessing (interpolation, resampling).

**License**

- This web application is licensed free of charge for academic use and we shall not be liable for any direct, indirect, incidental, or consequential damages resulting from the use of this web app. In addition, we are under no obligation to provide maintenance, support, updates, enhancements, or modifications.

""")