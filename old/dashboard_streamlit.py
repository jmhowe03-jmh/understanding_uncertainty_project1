import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st

# Load data
st.set_page_config(page_title="S&P 500 Analysis Dashboard", layout="wide")
df_companies = pd.read_csv('data/sp500_companies.csv')
df_index = pd.read_csv('data/sp500_index.csv')
df_index['Date'] = pd.to_datetime(df_index['Date'])
df_stocks = pd.read_csv('data/sp500_stocks.csv')


# --- Model Selection Selectbox ---
st.markdown("<h3 style='color:#f8f9fa;'>Choose Model</h3>", unsafe_allow_html=True)
model_type = st.selectbox(
    "",
    ("Kernel Regression", "Dummy Model"),
    index=0
)

y = df_index['S&P500'].values
x = np.arange(len(y))

if model_type == "Kernel Regression":
    lc_model = KernelReg([y], [x], var_type='c', reg_type='lc', bw=[20])
    y_pred = lc_model.fit(x)[0]
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    pred_label = 'Predicted S&P500'
else:
    # Dummy model: mean prediction
    y_pred = np.full_like(y, np.mean(y))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    pred_label = 'Dummy Model (Mean)'

# Line figure for True vs Predicted (static)
fig_true_vs_pred = px.line(
    x=df_index['Date'],
    y=[y, y_pred],
    labels={'x': 'Index', 'value': 'S&P500 Value', 'variable': 'Legend'},
    title='True vs Predicted S&P500 Values'
)
fig_true_vs_pred.update_traces(mode='lines')
fig_true_vs_pred.data[0].name = 'True S&P500'
fig_true_vs_pred.data[1].name = pred_label
fig_true_vs_pred.update_layout(xaxis_title='Date', yaxis_title='S&P500 Value')

# Markdown content from notebook cells (unchanged)
markdown_cells = [
    "### Data Description\n\nThis data set consists of three files, the first file(df_companies) has 502 rows, and 16 columns, the second file (df_index) has 2517 rows, and 2 colums, and the third file (df_stocks) has 1,891,536 rows and 8 columns.\n\nThe companies file has one row per company in the S&P 500 index. The columns are mostly descriptive providing information about each company, as well as some financial information and the weight of the company in the S&P 500 index.\n\nThe index file has one row per day and expands the time periord between December 2014 and December 2024. It records the market price of the S&P index on each day.\n\nThe final data set (df_stocks) has one row for each day for each company, and this is why there are so many rows. It has columns for the price of every stock on every day at different points during the trading window including the opening price and closing price, as well as the trading volume.\n\nHowever, this dataset is missing data on around 2/3 of the companies. This is a known issue, and it is discussed on kaggle Disussion Forum.",
    "### Data Provenance\nThis data set is collected by a kaggle user, named Larxel from the fedral reserve economic data (FRED) & yahoo finance. The data set used to be updated daily, but it has not been updated since December 2024. \n\nThis data is designed for people to use it to do stock market analysis, and this is why it is posted on kaggle.",
    "### Data Relevance & Modelling\n\nThe data that is most relevant to the project are going to be stock prices since the S&P 500 is not a stock, it does not have the price, but rather a value. This value is calculated based off weighted sum of all the underlying stocks. This is only computed once a day.\n\nFor individual stocks, we have five different prices for every day and every stock. The first price is the opening price which is the price the stock trades at the very moment the stock market opens, we also have the highest prices of everyday, and lowest prices of everyday, and closing price which is the price of the stock at the moment stock market closes.Finally, we have the adjusted closing price, which is the closing price after additional transcations such as divident payments that are accounted for.\n\nThe minimum stock price has gone down to $1829.08, and maxinum stock price was $6090.27. The average stock price has reached $3346.35. However, measuring the average stock price does not really make sense since it is over a such long time period, and the oversations are not independent.",
    "### Model Description\n\nLocal constant least squares regression is a non-parametric regression that looks at each point of the dependent variable only in relation to the points surrounding it. The number of points surrounding it is defined by the bandwidth parameter. The choice of bandwidth is very important to the success of this model.\n\nA key feature of this model is that the points around it are weighted with higher weights being assigned to points closest to the target point. The function used to assign weight is called the kernel. For a local least squares regression, multiple kernels are possible. \n\nAfter the points are weighted, the weighted average of all the points within the bandwidth is assigned for prediction for the target variable. The weighted average is also called the local constant, which is also what gives this model its name.\n\nWhile this model will not be difficult to code by hand due to its simplicity, but it is much easier to use statistical package such as one contained in the pyhton libray, statsmodels.\n\nThe stats model kernel regression chose the appropriate bandwidth automatically using least squares validation. A different choice of bandwidth might yield different results. In particular, a higher bandwidth will lead to a smoother curve, but a higher rmse. This is an example of bias and variance trade off where a higher rmse might be more likely to generalize for future points since small variations in stock prices from day to day will not be predictable. Using a small bandwidth will result small error, but will be overfit to minor and random pattern in the data.\n\nFor this analysis, I used the S&P 500 index values due to their completeness. To model individual stocks, I would need to check if they were included in the data that I downloaded which had a significant amount of missing data.",
    "### Data Processing & Model Fitting",
    "### Single Model Example",
    "### Bootstrapping the house down boots",
    "### Model Evaluation (p5)\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "### Conclusion (p6)",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
]

# --- Streamlit Layout ---
st.markdown("""
    <style>
    body, .main, .stApp {
        background: #23272b !important;
    }
    .block-style {
        background: #343a40 !important;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Roboto', sans-serif;
        font-size: 1.05rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
        margin-bottom: 20px;
        color: #f8f9fa !important;
    }
    .metric-style {
        display: inline-block;
        margin-left: 24px;
        font-family: 'Roboto', sans-serif;
        font-size: 1.1rem;
        color: #f8f9fa;
        background: #495057;
        border-radius: 6px;
        padding: 6px 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
    }
    h1, h2, h3, h4, h5, h6, .stTitle, .stHeader, .stSubheader {
        color: #f8f9fa !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("S&P 500 Analysis Dashboard")

# Markdown blocks (order preserved)
for i in range(6):
    st.markdown(markdown_cells[i], unsafe_allow_html=True)

st.subheader("True vs Predicted S&P500 Values")
st.plotly_chart(fig_true_vs_pred, use_container_width=True)
st.markdown(f'<div class="metric-style">Current model RMSE: {rmse:.3f}  MAE: {mae:.3f}</div>', unsafe_allow_html=True)

st.markdown(markdown_cells[6], unsafe_allow_html=True)


# Bootstrapping controls
n_boot = st.number_input("Number of bootstraps", min_value=1, max_value=10000, value=10, step=1)
if st.button("Run Bootstraps"):
    n = len(y)
    rng = np.random.default_rng(2)
    rmse_boot = np.zeros(n_boot)
    mae_boot = np.zeros(n_boot)
    for i in range(n_boot):
        sample = df_index.sample(n=n, replace=True, random_state=None)
        y_boot = sample['S&P500'].values
        x_boot = sample.index.values
        if model_type == "Kernel Regression":
            model_b = KernelReg([y_boot], [x_boot], var_type='c', reg_type='lc', bw=[20])
            try:
                yb_pred = model_b.fit(x)[0]
            except Exception:
                yb_pred = y_pred
        else:
            # Dummy model: mean prediction
            yb_pred = np.full_like(y, np.mean(y_boot))
        rmse_boot[i] = np.sqrt(np.mean((y - yb_pred) ** 2))
        mae_boot[i] = np.mean(np.abs(y - yb_pred))
    rmse_kde = ff.create_distplot([rmse_boot.tolist()], ["RMSE"], bin_size=0.1)
    mae_kde = ff.create_distplot([mae_boot.tolist()], ["MAE"], bin_size=0.1)
    st.subheader("RMSE Bootstrapped Distribution")
    st.plotly_chart(rmse_kde, use_container_width=True)
    st.subheader("MAE Bootstrapped Distribution")
    st.plotly_chart(mae_kde, use_container_width=True)
    # Dashboard figure with updated histograms
    fig_dashboard = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "True vs Predicted S&P500 Values",
            "RMSE Bootstrapped Distribution",
            "MAE Bootstrapped Distribution",
            "Company Market Cap vs Weight"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_index['Date'], y=y, mode='lines', name='True S&P500', line=dict(color='#636efa')),
        row=1, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_index['Date'], y=y_pred, mode='lines', name=pred_label, line=dict(color='#EF553B')),
        row=1, col=1
    )
    fig_dashboard.add_trace(
        go.Histogram(x=rmse_boot, name='RMSE', marker_color='rgb(31, 119, 180)', opacity=0.7, histnorm='probability density'),
        row=1, col=2
    )
    fig_dashboard.add_trace(
        go.Scatter(x=[rmse]*10, y=[0.0]*10, mode='markers',
                   marker=dict(color='red', symbol='x', size=10), name='RMSE (model)'),
        row=1, col=2
    )
    fig_dashboard.add_trace(
        go.Histogram(x=mae_boot, name='MAE', marker_color='rgb(31, 119, 180)', opacity=0.7, histnorm='probability density'),
        row=2, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=[mae]*10, y=[0.0]*10, mode='markers',
                   marker=dict(color='red', symbol='x', size=10), name='MAE (model)'),
        row=2, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(
            x=df_companies['Marketcap'],
            y=df_companies['Weight'],
            mode='markers',
            marker=dict(
                size=8,
                color=df_companies['Sector'].astype('category').cat.codes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sector')
            ),
            text=df_companies['Shortname'],
            name='Companies'
        ),
        row=2, col=2
    )
    fig_dashboard.update_layout(height=900, width=1200, title_text="S&P 500 Analysis Dashboard", showlegend=False)
    fig_dashboard.update_xaxes(title_text="Date", row=1, col=1)
    fig_dashboard.update_yaxes(title_text="S&P500 Value", row=1, col=1)
    fig_dashboard.update_xaxes(title_text="RMSE", row=1, col=2)
    fig_dashboard.update_yaxes(title_text="Density", row=1, col=2)
    fig_dashboard.update_xaxes(title_text="MAE", row=2, col=1)
    fig_dashboard.update_yaxes(title_text="Density", row=2, col=1)
    fig_dashboard.update_xaxes(title_text="Market Cap", row=2, col=2, type='log')
    fig_dashboard.update_yaxes(title_text="Index Weight", row=2, col=2)
    st.subheader("Full Dashboard")
    st.plotly_chart(fig_dashboard, use_container_width=True)

# Remaining markdown blocks
for i in range(7, len(markdown_cells)):
    st.markdown(markdown_cells[i], unsafe_allow_html=True)
