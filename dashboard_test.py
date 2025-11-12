import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
import seaborn as sns
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go

# Load data
df_companies = pd.read_csv('data/sp500_companies.csv')
df_index = pd.read_csv('data/sp500_index.csv')
df_index['Date'] = pd.to_datetime(df_index['Date'])
df_stocks = pd.read_csv('data/sp500_stocks.csv')

# Local constant kernel regression on S&P500 index (single fit at import)
y = df_index['S&P500'].values
x = np.arange(len(y))
lc_model = KernelReg([y], [x], var_type='c', reg_type='lc', bw=[20])
y_pred = lc_model.fit(x)[0]
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mae = np.mean(np.abs(y - y_pred))

# Line figure for True vs Predicted (static)
fig_true_vs_pred = px.line(
    x=df_index['Date'],
    y=[y, y_pred],
    labels={'x': 'Index', 'value': 'S&P500 Value', 'variable': 'Legend'},
    title='True vs Predicted S&P500 Values'
)
fig_true_vs_pred.update_traces(mode='lines')
fig_true_vs_pred.data[0].name = 'True S&P500'
fig_true_vs_pred.data[1].name = 'Predicted S&P500'
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

# Dash app
external_stylesheets = [
    'https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("S&P 500 Analysis Dashboard", className="display-4 text-center mb-4", style={
        'fontFamily': 'Roboto, sans-serif',
        'fontWeight': 700,
        'color': '#222',
        'letterSpacing': '1px',
        'background': 'linear-gradient(90deg, #e3ffe8 0%, #eaf6ff 100%)',
        'borderRadius': '12px',
        'padding': '24px 0',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
    }),

    html.Div([
        dcc.Markdown(cell, style={
            'marginBottom': '20px',
            'background': '#f8f9fa',
            'borderRadius': '8px',
            'padding': '16px',
            'fontFamily': 'Roboto, sans-serif',
            'fontSize': '1.05rem',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'
        })
        for cell in markdown_cells
    ], className="mb-4"),

    html.Div([
        html.Label("Number of bootstraps:", style={
            'fontWeight': 500,
            'marginRight': '10px',
            'fontFamily': 'Roboto, sans-serif',
            'fontSize': '1.1rem'
        }),
        dcc.Input(id='n-boot-input', type='number', min=1, max=10000, step=1, value=10, style={
            'marginRight': '10px',
            'width': '120px',
            'borderRadius': '6px',
            'border': '1px solid #ced4da',
            'padding': '6px 10px',
            'fontFamily': 'Roboto, sans-serif',
            'fontSize': '1rem'
        }),
        html.Button([
            html.I(className="fas fa-play me-2"), "Run Bootstraps"
        ], id='run-button', n_clicks=0, className="btn btn-success", style={
            'fontWeight': 600,
            'fontFamily': 'Roboto, sans-serif',
            'fontSize': '1rem',
            'padding': '6px 18px',
            'borderRadius': '6px',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.06)'
        }),
        html.Div("Current model RMSE: {:.3f}  MAE: {:.3f}".format(rmse, mae), style={
            'display': 'inline-block',
            'marginLeft': '24px',
            'fontFamily': 'Roboto, sans-serif',
            'fontSize': '1.1rem',
            'color': '#495057',
            'background': '#e9ecef',
            'borderRadius': '6px',
            'padding': '6px 14px',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'
        })
    ], className="d-flex align-items-center mb-4 justify-content-center gap-2"),

    html.H2("True vs Predicted S&P500 Values", className="h4 mt-4 mb-2 text-primary", style={
        'fontFamily': 'Roboto, sans-serif',
        'fontWeight': 600
    }),
    dcc.Graph(id='true-pred-graph', figure=fig_true_vs_pred, style={
        'background': '#fff',
        'borderRadius': '10px',
        'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
        'padding': '12px',
        'marginBottom': '32px'
    }),

    html.H2("RMSE Bootstrapped Distribution", className="h4 mt-4 mb-2 text-secondary", style={
        'fontFamily': 'Roboto, sans-serif',
        'fontWeight': 600
    }),
    dcc.Loading(dcc.Graph(id='rmse-kde-graph', style={
        'background': '#fff',
        'borderRadius': '10px',
        'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
        'padding': '12px',
        'marginBottom': '32px'
    })),

    html.H2("MAE Bootstrapped Distribution", className="h4 mt-4 mb-2 text-secondary", style={
        'fontFamily': 'Roboto, sans-serif',
        'fontWeight': 600
    }),
    dcc.Loading(dcc.Graph(id='mae-kde-graph', style={
        'background': '#fff',
        'borderRadius': '10px',
        'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
        'padding': '12px',
        'marginBottom': '32px'
    })),

    html.H2("Full Dashboard", className="h4 mt-4 mb-2 text-info", style={
        'fontFamily': 'Roboto, sans-serif',
        'fontWeight': 600
    }),
    dcc.Loading(dcc.Graph(id='dashboard-graph', style={
        'background': '#fff',
        'borderRadius': '10px',
        'boxShadow': '0 1px 8px rgba(0,0,0,0.06)',
        'padding': '12px',
        'marginBottom': '32px'
    }))
], style={
    'width': '98%',
    'maxWidth': '1400px',
    'margin': '0 auto',
    'background': '#f4f8fb',
    'borderRadius': '16px',
    'padding': '24px 0 32px 0',
    'boxShadow': '0 2px 16px rgba(0,0,0,0.07)'
})


@app.callback(
    Output('rmse-kde-graph', 'figure'),
    Output('mae-kde-graph', 'figure'),
    Output('dashboard-graph', 'figure'),
    Input('run-button', 'n_clicks'),
    State('n-boot-input', 'value')
)
def run_bootstraps(n_clicks, n_boot):
    # default if input missing
    if n_boot is None or n_boot < 1:
        n_boot = 10

    n = len(y)
    rng = np.random.default_rng(2)

    rmse_boot = np.zeros(n_boot)
    mae_boot = np.zeros(n_boot)

    # Bootstrapping loop
    for i in range(n_boot):
        sample = df_index.sample(n=n, replace=True, random_state=None)
        y_boot = sample['S&P500'].values
        x_boot = sample.index.values
        # fit on bootstrapped sample, predict on original x
        model_b = KernelReg([y_boot], [x_boot], var_type='c', reg_type='lc', bw=[20])
        try:
            yb_pred = model_b.fit(x)[0]
        except Exception:
            # fallback: if fitting failed, use original prediction
            yb_pred = y_pred
        rmse_boot[i] = np.sqrt(np.mean((y - yb_pred) ** 2))
        mae_boot[i] = np.mean(np.abs(y - yb_pred))

    # KDE / distribution figures using plotly.figure_factory
    rmse_kde = ff.create_distplot([rmse_boot.tolist()], ["RMSE"], bin_size=0.1)
    mae_kde = ff.create_distplot([mae_boot.tolist()], ["MAE"], bin_size=0.1)

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

    # 1. True vs Predicted S&P500 Values
    fig_dashboard.add_trace(
        go.Scatter(x=df_index['Date'], y=y, mode='lines', name='True S&P500', line=dict(color='#636efa')),
        row=1, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_index['Date'], y=y_pred, mode='lines', name='Predicted S&P500', line=dict(color='#EF553B')),
        row=1, col=1
    )

    # 2. RMSE Bootstrapped Distribution
    fig_dashboard.add_trace(
        go.Histogram(x=rmse_boot, name='RMSE', marker_color='rgb(31, 119, 180)', opacity=0.7, histnorm='probability density'),
        row=1, col=2
    )
    fig_dashboard.add_trace(
        go.Scatter(x=[rmse]*10, y=[0.0]*10, mode='markers',
                   marker=dict(color='red', symbol='x', size=10), name='RMSE (model)'),
        row=1, col=2
    )

    # 3. MAE Bootstrapped Distribution
    fig_dashboard.add_trace(
        go.Histogram(x=mae_boot, name='MAE', marker_color='rgb(31, 119, 180)', opacity=0.7, histnorm='probability density'),
        row=2, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=[mae]*10, y=[0.0]*10, mode='markers',
                   marker=dict(color='red', symbol='x', size=10), name='MAE (model)'),
        row=2, col=1
    )

    # 4. Company Market Cap vs Weight
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

    return rmse_kde, mae_kde, fig_dashboard


if __name__ == '__main__':
    app.run_server(debug=True)
