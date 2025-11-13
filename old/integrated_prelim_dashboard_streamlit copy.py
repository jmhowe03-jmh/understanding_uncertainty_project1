import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import random

# --- Data & Background ---
st.title("S&P 500 Analysis Dashboard")

st.header("Data & Background")

st.markdown("""
#### Data Description
This project uses three related datasets describing the S&P 500 index and its constituent companies. The S&P 500 is a market capitalization-weighted index of the 500 largest publicly listed companies trading in the United States. Market Capitalization-weighted means the share of the contributing stocks to the index is directly proportional to its relative market capitalization. The S&P 500 calculates a daily value (NAV) based on the weighted average of its components and is rebalanced quarterly. There are interquartile changes arising from new listings, delistings, and other corporate actions. Also, some stocks do not appear throughout the entire dataset as they may have experienced a valuation drawdown, listed publicly after 2010, or may have been taken private during the time period we are analyzing. The S&P 500 generally sees ~5% annual turnover and ~33% turnover per decade. Our data comes from Kaggle, and is compiled into an open source CSV format. The underlying stock prices and daily index value were collected from Yahoo Finance and the Federal Reserve Bank of St. Louis (FRED). The data was compiled primarily for research purposes as compiling historic stock price data can be a laborious and expensive task without access to proprietary databases and APIs. We are also given features of each company in the index including earnings before interest taxes and depreciation (EBITDA), market capitalization, weight to the index, geographic headquarters, among other factors.
""")

st.markdown("""
#### Modelling Phenomena & Rational
We are modelling how stock prices change over time. The efficient market hypothesis states that all publicly available information is reflected into a stock’s price instantly, however, there is a degree of “random” chaos that cannot be incorporated prior to manifestation. Stock prices are known to be highly volatile and subject to sporadic movement due to exogenous factors beyond business quality and economic factors. While we assume that stock price movement is an independent occurrence, in reality stocks across various sectors are known to be correlated in their price movements. This is the underlying intuition behind a stock’s Beta -- or its relative volatility to the broader market’s movement. 
""")

st.markdown("""
#### Data Exploration
""")

# --- Data Loading ---
sp_comp = pd.read_csv('data/sp500_companies.csv')
sp_index = pd.read_csv('data/sp500_index.csv')
sp_stocks = pd.read_csv('data/sp500_stocks.csv')

# --- Data Exploration Plots ---
# S&P 500 with moving averages
windows = [30, 90, 180, 365]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sp_index["Date"],
    y=sp_index["S&P500"],
    mode='lines',
    name="S&P 500",
    line=dict(color="black", width=1)
))
for w in windows:
    avg = sp_index["S&P500"].rolling(window=w, center=True, min_periods=w//2).mean()
    fig.add_trace(go.Scatter(
        x=sp_index["Date"],
        y=avg,
        mode='lines',
        name=f"{w}-day Moving Avg",
        line=dict(width=2)
    ))
fig.update_layout(
    title="S&P 500 Averages with bandwidths",
    xaxis_title="Date",
    yaxis_title="S&P 500",
    legend_title="Legend",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.pie(sp_comp, values='Marketcap', names='Sector', title='Marketcap by Sector')
st.plotly_chart(fig2, use_container_width=True)

# --- Models Section ---
n_cols = 3
states = ['Bear', 'Bull', 'Stable']

# --- Model Section with Segmented Button ---
st.header("Models")
model_choice = st.selectbox(
    "Select Model:",
    ("Model 1: Least Constant Squares Regression", "Model 2: Markov Chain Model"),
    index=0,
    key="model_segmented_button"
)

if model_choice == "Model 1: Least Constant Squares Regression":
    st.markdown("### Model 1: Least Constant Squares Regression")
    st.markdown("#### Model Description")
    st.markdown("""
Local constant least squares regression is a non-parametric regression that looks at each point of the dependent variable only in relation to the points surrounding it. The number of points surrounding it is defined by the bandwidth parameter. The choice of bandwidth is very important to the success of this model.

A key feature of this model is that the points around it are weighted with higher weights being assigned to points closest to the target point. The function used to assign weight is called the kernel. For a local least squares regression, multiple kernels are possible. 

After the points are weighted, the weighted average of all the points within the bandwidth is assigned for prediction for the target variable. The weighted average is also called the local constant, which is also what gives this model its name.

While this model will not be difficult to code by hand due to its simplicity, but it is much easier to use statistical package such as one contained in the python libray, statsmodels.
""")
    st.markdown("#### Singular Example (S&P500)")
    df_companies = pd.read_csv('data/sp500_companies.csv')
    df_index = pd.read_csv('data/sp500_index.csv')
    df_index['Date'] = pd.to_datetime(df_index['Date'])
    df_stocks = pd.read_csv('data/sp500_stocks.csv')
    y = df_index['S&P500']
    x = np.arange(len(y))
    lc_model = KernelReg([y],[x],var_type='c', reg_type='lc',bw = [20])
    y_pred = lc_model.fit(x)[0]
    rmse = np.sqrt(np.mean((y-y_pred)**2))
    mae= np.mean(np.abs(y-y_pred))
    fig3 = px.line(
        x=df_index['Date'], 
        y=[y, y_pred], 
        labels={'x': 'Index', 'value': 'S&P500 Value', 'variable': 'Legend'},
        title='True vs Predicted S&P500 Values'
    )
    fig3.update_traces(mode='lines')
    fig3.update_layout(
        legend=dict(
            title='',
            itemsizing='constant',
            traceorder='normal'
        ),
        xaxis_title='Date',
        yaxis_title='S&P500 Value'
    )
    fig3.data[0].name = 'True S&P500'
    fig3.data[1].name = 'Predicted S&P500'
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Bootstrapping & Evaluation")
    n_boot = 10
    rmse_boot = np.zeros(n_boot)
    mae_boot = np.zeros(n_boot)
    n = len(y)
    rng = np.random.default_rng(2)
    for i in range(n_boot):
        sample = df_index.sample(n=n, replace=True, random_state=None)
        y_boot = sample['S&P500'].values
        x_boot = sample.index.values
        model_b = KernelReg([y_boot], [x_boot], var_type='c', reg_type='lc', bw=[20])
        yb_pred = model_b.fit(x)[0]
        rmse_boot[i] = np.sqrt(np.mean((y - yb_pred) ** 2))
        mae_boot[i] = np.mean(np.abs(y - yb_pred))
    rmse_kde = ff.create_distplot(hist_data=[rmse_boot], group_labels=["RMSE"], bin_size=0.1)
    mse_kde = ff.create_distplot(hist_data=[mae_boot], group_labels=["MAE"], bin_size=0.1)
    st.plotly_chart(rmse_kde, use_container_width=True)
    st.plotly_chart(mse_kde, use_container_width=True)

elif model_choice == "Model 2: Markov Chain Model":
    st.markdown("### Model 2: Markov Chain Model")
    st.markdown("#### Model Description")
    st.markdown("""
To model the likelihood of bearish, bullish, or stable days, we split our dataset into a training set (up to 2020) and a test set (2020–2024). For each stock, we calculate the daily percent change in adjusted closing prices and classify each day into one of three states:
1. Bear: negative percent change (market goes down)
2. Bull: positive percent change (market goes up)
3. Stable: percent change within a defined threshold
""")
    st.markdown("#### Individual Stock Level")
    st.markdown("This classification allows us to generate state sequences for each stock across the years, forming the basis for individual stock transition matrices, which capture the probability of moving from one state to another.")

    sp_stocks = sp_stocks.sort_values(['Date','Symbol'])
    sp_stocks['pct_change'] = sp_stocks.groupby('Symbol')['Adj Close'].pct_change()
    sp_stocks = sp_stocks.dropna(subset=['pct_change'])

    st.markdown("##### Threshold Decision")
    st.markdown("The threshold defines the range considered Stable. We experimented with values from 0.001 to 0.2. A threshold of 0.001 was small, barely reflecting stable days, while 0.2 was too large, resulting in most days being classified as stable. A threshold of 0.002 (0.2%) was chosen. This captures minor flat days as stable while keeping Bull and Bear proportions realistic.")

    threshold = 0.002
    def classify_change(x):
        if x > threshold:
            return 'Bull'
        elif x < -threshold:
            return 'Bear'
        else:
            return 'Stable'
    sp_stocks['State'] = sp_stocks['pct_change'].apply(classify_change)
    sp_counts = sp_stocks['State'].value_counts(normalize=True).reset_index()
    sp_counts.columns = ['State', 'proportion']
    fig4 = px.pie(sp_counts, values='proportion', names='State', title='Proportion of Market States')
    st.plotly_chart(fig4, use_container_width=True)

    ticker = sp_stocks[sp_stocks['Date'] < '2020-01-01']['Symbol'].unique()
    train = sp_stocks[sp_stocks['Date'] < '2020-01-01']
    transition_matrices = {}
    for t in ticker:
        ticker_state = train[train['Symbol'] == t].sort_values('Date')['State'].values
        transition_matrices[t] = pd.crosstab(ticker_state[:-1], ticker_state[1:], normalize='index')

    st.markdown("##### Interpretation")
    st.markdown("From the value counts of our states, we observe that daily movements are almost evenly split between Bull and Bear days, with Stable days accounting for around 13%. Examining a single stock like CCL, we find that it is 40% likely to move from Bear to Bear, 46% from Bear to Bull, and 13% from Bear to Stable, reflecting the randomness of stock movements.")

    st.markdown("#### Sector Level Analysis")
    st.markdown("After computing individual stock transition matrices, we aggregate them to the sector level.")
    sp_sector = train.merge(sp_comp[['Symbol','Sector','Marketcap','Weight']], how='left', on='Symbol')
    sector_groups = sp_sector[['Symbol','Sector']].drop_duplicates()
    state_counts = (
        sp_sector.groupby(['Sector', 'State'])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )
    state_counts_reset = state_counts.reset_index().melt(id_vars='Sector', var_name='State', value_name='Proportion')
    fig5 = px.bar(
        state_counts_reset,
        x='Sector',
        y='Proportion',
        color='State',
        title='Proportion of Bull vs Bear Days by Sector',
        barmode='stack',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig5.update_layout(
        xaxis_title='Sector',
        yaxis_title='Proportion',
        legend_title='State'
    )
    st.plotly_chart(fig5, use_container_width=True)

    # --- Sector Transition Matrices ---
    states = ['Bear', 'Bull', 'Stable']
    sectors = sector_groups['Sector'].unique()
    sector_transitions = {}
    for sector in sectors:
        symbol_sector = sector_groups[sector_groups['Sector'] == sector]['Symbol']
        matrics, weights = [], []
        for t in symbol_sector:
            if t in transition_matrices:
                matrics.append(transition_matrices[t])
                weight = sp_comp.loc[sp_comp['Symbol'] == t, 'Marketcap'].values[0]
                weights.append(weight)
        if matrics:
            matrics = [m.values if isinstance(m, pd.DataFrame) else m for m in matrics]
            weighted_sum = sum(m*w for m,w in zip(matrics, weights))
            avg_matrix = weighted_sum / sum(weights)
            sector_transitions[sector] = pd.DataFrame(avg_matrix, index=states, columns=states)

    n_sectors = len(sectors)
    n_cols = 3
    n_rows = int(np.ceil(n_sectors / n_cols))
    fig6 = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=sectors,
        horizontal_spacing=0.08, vertical_spacing=0.12
    )
    for idx, sector in enumerate(sectors):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        matrix = sector_transitions[sector].values
        heatmap = go.Heatmap(
            z=matrix,
            x=states,
            y=states,
            colorscale='tealrose',
            zmin=0, zmax=0.5,
            colorbar=dict(title='Probability'),
            showscale=(col == n_cols)
        )
        fig6.add_trace(heatmap, row=row, col=col)
    fig6.update_layout(
        height=300 * n_rows,
        width=350 * n_cols,
        title_text="Sector Transition Matrices",
        showlegend=False
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("#### Simulation and Validation")
    st.markdown("Using the test dataset (2020–2024), we simulate sector-level state sequences based on the last observed state in the training set. The simulated transition matrices are then compared to the actual transition matrices for the same period.")

    def simulate_sector(start_state, transition_matrix, days):
        states = ['Bear','Bull','Stable']
        sequence = [start_state]
        for _ in range(days-1):
            probs = transition_matrix.loc[sequence[-1]].values
            next_state = random.choices(states, weights=probs, k=1)[0]
            sequence.append(next_state)
        return sequence

    # Dictionary to store simulated sector states
    simulated_sector_states = {}
    test  = sp_stocks[sp_stocks['Date'] >= '2020-01-01']
    test = test.merge(sp_comp[['Symbol','Sector','Marketcap','Weight']], how='left', on='Symbol')
    train = train.merge(sp_comp[['Symbol','Sector','Marketcap','Weight']], how='left', on='Symbol')

    for sector in sectors:
        last_state = train[train['Sector']==sector].sort_values('Date')['State'].iloc[-1]
        days = test[test['Sector']==sector]['Date'].nunique()
        simulated_sector_states[sector] = simulate_sector(last_state, sector_transitions[sector], days)
    actual_states = test[['Sector', 'State', 'Date']].copy()
    rows = []
    for sector in sectors:
        sim_counts = pd.Series(simulated_sector_states[sector]).value_counts(normalize=True)
        actual_counts = actual_states[actual_states['Sector'] == sector]['State'].value_counts(normalize=True)
        for state in ['Bear', 'Bull', 'Stable']:
            rows.append({
                'sector': sector,
                'state': state,
                'simulated proportion': sim_counts.get(state, 0.0),
                'actual proportion': actual_counts.get(state, 0.0)
            })
    sector_state_df = pd.DataFrame(rows)
    st.dataframe(sector_state_df)

    st.markdown("#### Converting States to Price Movements")
    st.markdown("We convert the States to Price movement, by assigning percentage increase or decrease to each state. Starting from the last training state, we simulate multiple price paths and compute the mean predicted price along with a 95% confidence interval.")

    states = ['Bear', 'Bull', 'Stable']
    state_to_return = {'Bear': -0.015, 'Stable': 0, 'Bull': 0.015}
    test_stock = train['Symbol'].unique()[0]
    last_state = train[train['Symbol']==test_stock].iloc[-1]['State']
    transition_matrix = transition_matrices[test_stock]
    n_days = len(test[test['Symbol']==test_stock])
    simulations = 1000
    simulated_prices = np.zeros((simulations, n_days))
    initial_price = train[train['Symbol']==test_stock].iloc[-1]['Adj Close']
    for i in range(simulations):
        price = initial_price
        state = last_state
        for day in range(n_days):
            probs = transition_matrix.loc[state].values
            state = np.random.choice(states, p=probs)
            price *= 1 + state_to_return[state]
            simulated_prices[i, day] = price
    mean_pred = simulated_prices.mean(axis=0)
    lower = np.percentile(simulated_prices, 2.5, axis=0)
    upper = np.percentile(simulated_prices, 97.5, axis=0)
    actual_prices = test[test['Symbol']==test_stock]['Adj Close'].values
    train_prices = train[train['Symbol']==test_stock]['Adj Close'].values
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=list(range(len(train_prices))),
        y=train_prices,
        mode='lines',
        name='Training Data',
        line=dict(color='blue')
    ))
    fig7.add_trace(go.Scatter(
        x=list(range(len(train_prices), len(train_prices) + len(actual_prices))),
        y=actual_prices,
        mode='lines',
        name='Actual Test Data',
        line=dict(color='green')
    ))
    fig7.add_trace(go.Scatter(
        x=list(range(len(train_prices), len(train_prices) + len(mean_pred))),
        y=mean_pred,
        mode='lines',
        name='Mean Prediction',
        line=dict(color='red')
    ))
    fig7.add_trace(go.Scatter(
        x=list(range(len(train_prices), len(train_prices) + len(mean_pred))),
        y=upper,
        mode='lines',
        name='95% Confidence Upper',
        line=dict(width=0),
        showlegend=False
    ))
    fig7.add_trace(go.Scatter(
        x=list(range(len(train_prices), len(train_prices) + len(mean_pred))),
        y=lower,
        mode='lines',
        name='95% Confidence Lower',
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(width=0),
        showlegend=True
    ))
    fig7.update_layout(
        title=f'{test_stock} Stock Price: Markov Chain Prediction',
        xaxis_title='Days',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_white'
    )
    st.plotly_chart(fig7, use_container_width=True)

st.header("Results & Conclusions")
st.markdown("""
We observe several properties of the training data reflected in our simulated price paths. For example, simulated prices never fall below $0, and the trajectories broadly mirror the long-run trend of the training data (steady upward drift). This is expected given the train-test split and the fact that the Markov chain only models transitions between discretized return states. However, these simulations should not be interpreted as credible forecasts. The model has high uncertainty because it does not incorporate macroeconomic conditions, firm-specific information, or any structural drivers of returns. The transition probabilities are derived solely from historical day-to-day movements, so the model cannot account for exogenous shocks or shifts in market regimes. Moreover, the Markov property imposes memory-lessness: only the most recent state influences the next movement. As a result, the model cannot capture persistent dynamics like momentum, volatility clustering, or mean reversion that extend beyond a single period. The outputs therefore reflect historical transition patterns, not a realistic prediction of future prices.

The primary drawback of our model is that Markov chains are memoryless. Our model does not mimic real world exogenous factors that influence stock prices, and thereby the S&P 500. Our model assumes that the only factors that determine a future price movement are the current price and its sector, whereas in the real world, sentiment, volatility, and market events all feed into a stock’s price movement. For future work on this project, we would want to explore a parametric model that incorporates more inputs in the form of parameters that allow for memory in the model -- this would better capture a company’s qualitative and quantitative measures of performance as well as market sentiments / trends to project future price movements. For instance, transformers may allow for sentiment in the form of news headlines to be factored into future price estimations. A more robust model would allow for collection of inputs beyond just current price, which our current model does not permit. Examples of parameters include trading volume, moving average of price, company earnings transcripts, and others that would present more relevant data than just current price.
""")
