# -prophet-challenge

# Install the required libraries
!pip install prophet

# Import the required libraries and dependencies
import pandas as pd
from prophet import Prophet
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Store the data in a Pandas DataFrame
# Set the "Date" column as the Datetime Index.

df_mercado_trends = pd.read_csv(
    "https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/google_hourly_search_trends.csv",
    index_col='Date',
    parse_dates=True
).dropna()

# Review the first and last five rows of the DataFrame
display(df_mercado_trends.head())
display(df_mercado_trends.tail())

# Review the data types of the DataFrame using the info function
df_mercado_trends.info()

# Slice the DataFrame to just the month of May 2020
# Slice the DataFrame to just the month of May 2020
#sliced_df = df_mercado_trends[2020-5-1:2020-5-31]
# Example: Slice data between two dates
sliced_df=df_mercado_trends.loc['2020-5-1':'2020-5-31']['Search Trends'].plot(figsize=(10,6))

# Plot to visualize the data for May 2020
sliced_df.plot()

# Calculate the sum of the total search traffic for May 2020
total_may_2020_traffic = df_mercado_trends.loc['2020-5-1':'2020-5-31']['Search Trends'].sum()

# View the traffic_may_2020 value
#total_may_2020_traffic = sliced_df['Search Trends'].sum()
total_may_2020_traffic

# Calcluate the monhtly median search traffic across all months
# Group the DataFrame by index year and then index month, chain the sum and then the median functions
df_mercado_trends['Month'] = df_mercado_trends.index.month
df_mercado_trends['Year'] = df_mercado_trends.index.year
median_monthly_traffic = df_mercado_trends.groupby(['Year', 'Month'])['Search Trends'].sum().median()

# View the median_monthly_traffic value
median_monthly_traffic

# Compare the search traffic for the month of May 2020 to the overall monthly median value
#total_may_traffic_2020/median_monthly_traffic
comparison_value = total_may_2020_traffic / median_monthly_traffic
comparison_value
#print(f"Total search traffic for May 2020: {total_may_traffic_2020}")
#print(f"Monthly median search traffic across all months: {median_monthly_traffic}")
#print(f"Search traffic for May 2020 divided by the monthly median: {total_may_traffic_2020/median_monthly_traffic}")

# Group the hourly search data to plot the average traffic by the day of week, using `df.index.hour`
hourly_traffic = df_mercado_trends.groupby(df_mercado_trends.index.hour)['Search Trends'].mean()

plt.figure(figsize=(8, 5))
plt.plot(hourly_traffic.index, hourly_traffic.values)
plt.title('Average Search Traffic by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Search Traffic')
plt.xticks(range(0,25, 5))
plt.legend('Search Trends', fontsize=15)
plt.grid(False)
plt.show()

# Group the hourly search data to plot the average traffic by the day of week, using `df.index.isocalendar().day`.
import matplotlib.pyplot as plt
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_traffic = df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().day)['Search Trends'].mean()

plt.figure(figsize=(10, 6))
plt.plot(daily_traffic.index, daily_traffic.values)
plt.title('Average Search Traffic by Day of Week')
plt.xlabel('Day')
#plt.ylabel('Average Search Traffic')
#plt.xticks(1,7,1)
plt.grid(False)
plt.show()

# Group the hourly search data to plot the average traffic by the week of the year using `df.index.isocalendar().week`.

avg_weekly_traffic = df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().week)['Search Trends'].mean()

plt.figure(figsize=(10, 6))
plt.plot(avg_weekly_traffic.index, avg_weekly_traffic.values)
#plt.title('Average Search Traffic by Week of the Year')
plt.xlabel('Week')
#plt.ylabel('Average Search Traffic')
plt.grid(False)
plt.show()

# Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the Datetime Index.
df_mercado_stock = pd.read_csv(
    "https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/mercado_stock_price.csv",
    index_col="date",
    parse_dates=True
).dropna()

# View the first and last five rows of the DataFrame
display(df_mercado_stock.head())
display(df_mercado_stock.tail())

# Visualize the closing price of the df_mercado_stock DataFrame
plt.figure(figsize=(10,6))
plt.plot(df_mercado_stock.index, df_mercado_stock['close'])
plt.title('Closing Price of Mercado Stock')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend('Close', fontsize=10)
plt.xticks(rotation=45)
plt.show()

# Concatenate the df_mercado_stock DataFrame with the df_mercado_trends DataFrame
# Concatenate the DataFrame by columns (axis=1), and drop and rows with only one column of data
#mercado_stock_trends_df = pd.concat([df_mercado_stock, df_mercado_trends], axis=1).dropna()

mercado_stock_trends_df = pd.concat([df_mercado_stock[['close']], df_mercado_trends[['Search Trends']]], axis=1).dropna()


# View the first and last five rows of the DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())

# For the combined dataframe, slice to just the first half of 2020 (2020-01 through 2020-06)
first_half_2020 = mercado_stock_trends_df.loc['2020-01-01' : '2020-06-30']
# View the first and last five rows of first_half_2020 DataFrame
display(first_half_2020.head())
display(first_half_2020.tail())

# Visualize the close and Search Trends data
# Plot each column on a separate axes using the following syntax
# `plot(subplots=True)`

first_half_2020[['close', 'Search Trends']].plot(subplots=True, figsize=(9, 5))
plt.show()

# Create a new column in the mercado_stock_trends_df DataFrame called Lagged Search Trends
# This column should shift the Search Trends information by one hour
mercado_stock_trends_df['lagged_search_trends'] = mercado_stock_trends_df['Search Trends'].shift(1)

# View the first and last five rows of the new DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())

# Create a new column in the mercado_stock_trends_df DataFrame called Stock Volatility
# This column should calculate the standard deviation of the closing stock price return data over a 4 period rolling window
mercado_stock_trends_df['stock_volatility'] = mercado_stock_trends_df['close'].pct_change().rolling(window=4).std()

# View the first and last five rows of the DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())

# Visualize the stock volatility
mercado_stock_trends_df['stock_volatility'].plot()

# Create a new column in the mercado_stock_trends_df DataFrame called Hourly Stock Return
# This column should calculate hourly return percentage of the closing price
mercado_stock_trends_df['Hourly Stock Return'] = pd.DataFrame(mercado_stock_trends_df['close'].pct_change())

# View the first and last five rows of the DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())

# View the first and last five rows of the mercado_stock_trends_df DataFrame
mercado_stock_trends_df.head()
mercado_stock_trends_df.tail()

# Construct correlation table of Stock Volatility, Lagged Search Trends, and Hourly Stock Return
mercado_stock_trends_df[['stock_volatility', 'lagged_search_trends', 'Hourly Stock Return']].corr()


# Using the df_mercado_trends DataFrame, reset the index so the date information is no longer the index
#mercado_trends = df_mercado_trends.reset_index()

#mercado_prophet_df = df_mercado_trends.reset_index()
mercado_prophet_df = df_mercado_trends.reset_index()

# Label the columns ds and y so that the syntax is recognized by Prophet
#df_mercado_trends.columns = ['ds', 'y']

mercado_prophet_df = mercado_prophet_df.rename(columns={'Date': 'ds', 'Search Trends': 'y'})
mercado_prophet_df = mercado_prophet_df[['ds', 'y']]
# Drop an NaN values from the prophet_df DataFrame
mercado_prophet_df = mercado_prophet_df.dropna()


# View the first and last five rows of the mercado_prophet_df DataFrame
display(mercado_prophet_df.head())
display(mercado_prophet_df.tail())

# Call the Prophet function, store as an object
model_mercado_trend = Prophet()

# Fit the time-series model.
model_mercado_trend.fit(mercado_prophet_df)

# Create a future dataframe to hold predictions
# Make the prediction go out as far as 2000 hours (approx 80 days)
future_mercado_trends = model_mercado_trend.make_future_dataframe(periods=2000, freq='H')

# View the last five rows of the future_mercado_trends DataFrame
display(future_mercado_trends.tail())

# Make the predictions for the trend data using the future_mercado_trends DataFrame
forecast_mercado_trends = model_mercado_trend.predict(future_mercado_trends)

# Display the first five rows of the forecast_mercado_trends DataFrame
display(forecast_mercado_trends.head())

# Plot the Prophet predictions for the Mercado trends data
model_mercado_trend.plot(forecast_mercado_trends)

# Set the index in the forecast_mercado_trends DataFrame to the ds datetime column
forecast_mercado_trends = forecast_mercado_trends.set_index('ds')

# View the only the yhat,yhat_lower and yhat_upper columns from the DataFrame
forecast_mercado_trends[['yhat', 'yhat_lower', 'yhat_upper']]
#forecast_mercado_trends.head()

# From the forecast_mercado_trends DataFrame, plot the data to visualize
#  the yhat, yhat_lower, and yhat_upper columns over the last 2000 hours
forecast_mercado_trends[['yhat', 'yhat_lower', 'yhat_upper']].iloc[-2000:,:].plot()

# Reset the index in the forecast_mercado_trends DataFrame
forecast_mercado_trends = forecast_mercado_trends.reset_index()

# Use the plot_components function to visualize the forecast results
# for the forecast_mercado_trends DataFrame
model_mercado_trend.plot_components(forecast_mercado_trends)

