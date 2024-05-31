import streamlit as st
from datetime import date


import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from stocknews import StockNews
from datetime import datetime, timedelta

# Download NLTK data for sentiment analysis
import nltk
nltk.download('vader_lexicon')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Investment Value Forecast')

# Allow users to input a custom stock symbol
selected_stock = st.text_input('Enter stock symbol (e.g., AAPL):')

# Validate if the user has entered a stock symbol
if not selected_stock:
    st.warning('Please enter a valid stock symbol.')
    st.stop()

# Function to get additional stock details
def get_stock_details(ticker):
    stock_info = yf.Ticker(ticker).info
    return stock_info

# Get stock details
stock_details = get_stock_details(selected_stock)

# Display stock details table
st.subheader('Stock Details')

st.write("Full Name:", stock_details.get("longName", "N/A"))
st.write("Country of Origin:", stock_details.get("country", "N/A"))
st.write("Currency:", stock_details.get("currency", "N/A"))
st.write("Registered at:", stock_details.get("exchange", "N/A"))

# Function to get news sentiment
def get_news_sentiment(api_key, company_name):
    endpoint = "https://newsapi.org/v2/everything"
    params = {
        'q': company_name,
        'apiKey': api_key,
        'pageSize': 20,  # Number of articles to fetch
        'language': 'en',
    }

    response = requests.get(endpoint, params=params)
    news_data = response.json()

    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for article in news_data['articles']:
        title = article['title']
        sentiment_score = sid.polarity_scores(title)['compound']
        sentiment_scores.append(sentiment_score)

    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return average_sentiment

# News sentiment analysis
api_key = "5fc017437faf47c18358807d776f58b9"
news_sentiment = get_news_sentiment(api_key, stock_details.get("longName"))
st.write(f'Average News Sentiment: {news_sentiment}')




@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.iloc[::-1])  # Reverse the order of the dataframe

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Predict forecast with Prophet considering news sentiment
df_train_with_news = data[['Date','Close']]
df_train_with_news = df_train_with_news.rename(columns={"Date": "ds", "Close": "y"})
df_train_with_news['news_sentiment'] = news_sentiment

m_with_news = Prophet()
m_with_news.add_regressor('news_sentiment')
m_with_news.fit(df_train_with_news)
future_with_news = m_with_news.make_future_dataframe(periods=period)
future_with_news['news_sentiment'] = news_sentiment
forecast_with_news = m_with_news.predict(future_with_news)


# toggle on
on = st.toggle('Activate for considering current news in prediction')
if on:
        st.write('Feature activated!')
        # Show and plot forecast with news sentiment
        st.subheader('Forecast data with News Sentiment')
        st.write(forecast_with_news[['ds','yhat','yhat_lower','yhat_upper']].tail())

        st.write(f'Forecast plot for {n_years} years with News Sentiment')
        fig1_with_news = plot_plotly(m_with_news, forecast_with_news)
        st.plotly_chart(fig1_with_news)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


Trends_of_Stock,Additional_Details,News = st.tabs(["Trends","Additional Details","Top News"])


with Trends_of_Stock:
    st.header("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

with Additional_Details:

    Past_24_hrs_data,Specific_Period,Price_Movements = st.tabs(["Past 24hrs Data","Specific Period","Price Movements"])

    with Past_24_hrs_data:
        st.header('Last 24 hours data')

        # Function to load past 24 hours of stock data with 1-hour interval
        def load_past_24_hours_data(ticker):
            end_datet = datetime.today()
            start_datet = end_datet - timedelta(days=1)

        # Download historical data with 1-hour interval
            datat = yf.download(ticker, start=start_datet, end=end_datet, interval='1h')
            return datat

        # Load past 24 hours of stock data
        datat = load_past_24_hours_data(selected_stock)
        st.write(datat)

    with Specific_Period:
        st.header('Analysis for Specific Period')

        on = st.toggle('Activate to enter')

        if on:
          st.write('Feature activated!')
          starts=st.sidebar.date_input('Start Date')
          ends=st.sidebar.date_input('End Date')
          datas=yf.download(selected_stock,start=starts,end=ends)
          fig=px.line(datas,x=datas.index,y=datas['Adj Close'],title=selected_stock)
          st.plotly_chart(fig)


    with Price_Movements:
        st.header('Price Movements')
        data2=data
        data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
        data2.dropna(inplace = True)
        st.write(data2.iloc[::-1])
        annual_return =data2['% Change'].mean()*252*100
        st.write('Annual Return is ',annual_return,'%')

    
with News:
    st.header(f'News of {stock_details.get("longName")}')
    def get_news(api_key, query):
        endpoint = "https://newsapi.org/v2/everything"
        params = {
            'q': query,  # Query for the company name or ticker symbol
            'apiKey': api_key,
            'pageSize': 20,  # Number of articles to fetch
            'language': 'en',
            'sortBy': 'publishedAt',
        }

        response = requests.get(endpoint, params=params)
        news_data = response.json()
    
        return news_data['articles']

    # Example usage
    api_key = '5fc017437faf47c18358807d776f58b9'
  # You can use the company name or related keywords
    articles = get_news(api_key, stock_details.get("longName"))

# Print the title and description of each article
    for article in articles:
        st.write("Title:", article['title'])
        st.write("Description:", article.get('description', 'No description available'))
        st.write("URL:", article['url'])
        st.write()



