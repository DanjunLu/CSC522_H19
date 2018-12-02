# -------------------------------------------------------------- Part 1

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



# ----------------------------------------------------------- Part 2
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()

# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()





# ------------------------------------------------------ Part 3
(market_train_df, news_train_df) = env.get_training_data()
import numpy as np 
import pandas as pd
import re
import pandasql as ps
import datetime as dt
import gc
import math
import plotly.offline as py
py.init_notebook_mode(connected=True)





# ------------------------------------------------------ Part 4
def top_std_plot(market_train_df, scale=1):
    market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
    grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
    g = grouped.sort_values(('price_diff', 'std'), ascending=False).head(10)
    g['min_text'] = 'Maximum price drop: ' + (-1 * g['price_diff']['min']).astype(str)

    trace = go.Scatter(
        x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = g['price_diff']['std'].values,
        mode='markers',
        marker=dict(
            size = g['price_diff']['std'].values * scale,
            color = g['price_diff']['std'].values,
            colorscale='Portland',
            showscale=True
        ),
        text = g['min_text'].values
        #text = f"Maximum price drop: {g['price_diff']['min'].values}"
        #g['time'].dt.strftime(date_format='%Y-%m-%d').values
    )

    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Top 10 months of daily price chance standard deviation',
        hovermode= 'closest',
        yaxis=dict(
            title= 'standard deviation',
            ticklen= 5,
            gridwidth= 2,
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')

top_std_plot(market_train_df, 1)







# ---------------------------------------------------------- Part 5
print(market_train_df.sort_values('price_diff').head(10)[['assetName', 'close', 'open']])

market_train_df['CloseOpenRatio'] = np.abs(market_train_df['close'] / market_train_df['open'])


market_train_df['mean_open'] = market_train_df.groupby('assetName')['open'].transform('mean')
market_train_df['mean_close'] = market_train_df.groupby('assetName')['close'].transform('mean')

# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.
for i, row in market_train_df.loc[market_train_df['CloseOpenRatio'] >= 2].iterrows():
    if np.abs(row['mean_open'] - row['open']) > np.abs(row['mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['mean_open']
    else:
        market_train_df.iloc[i,4] = row['mean_close']
        
for i, row in market_train_df.loc[market_train_df['CloseOpenRatio'] <= 0.5].iterrows():
    if np.abs(row['mean_open'] - row['open']) > np.abs(row['mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['mean_open']
    else:
        market_train_df.iloc[i,4] = row['mean_close']
        
print(market_train_df.sort_values('price_diff').head(10)[['assetName', 'close', 'open']])
        

top_std_plot(market_train_df, 5)






# ---------------------------------------------------------- Part 6
data = []
for i in [0.05, 0.25, 0.5, 0.75, 0.95]:
    price_df = market_train_df.groupby('time')['returnsOpenNextMktres10'].quantile(i).reset_index()

    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['returnsOpenNextMktres10'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of returnsOpenNextMktres10 by quantiles",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = '10 Day Open Returns'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=data, layout=layout), filename='basic-line')



stocks = ['GOOG.O', 'YHOO.O', 'AAPL.O', 'MSFT.O', 'FB.O']

market_train_df = market_train_df.loc[market_train_df['assetCode'] in stocks]
news_train_df = news_train_df.loc[news_train_df['assetCode'] in stocks]

market_train_df = market_train_df.loc[market_train_df['time'] >= '2015-01-01 22:00:00+0000']
news_train_df = news_train_df.loc[news_train_df['time'] >= '2015-01-01 22:00:00+0000']





# ---------------------------------------------------------- Part 7
def manipulate_news(df):
    
    df[["Code_N","Code_O","Code"]] = df["assetCodes"].str.extract(r"'([a-zA-Z]{0,6}\.N)'|'([a-zA-Z]{0,6}\.O)'|'([a-zA-Z]{0,6})'", expand=True)
    temp_df = pd.DataFrame(df["Code_N"].combine_first(df["Code_O"]).combine_first(df["Code"]))
    df[["assetCode"]] = temp_df
    del temp_df
    gc.collect()
    
    # Variables that will not be included in the model.
    drop_names = ['sourceTimestamp', 'firstCreated', 'sourceId', 'headline', 'takeSequence', 'provider', 
                  'subjects', 'audiences', 'bodySize', 'companyCount', 'headlineTag', 'sentenceCount', 
                  'wordCount', 'assetName', 'firstMentionSentence', 'assetCodes', 'Code_N', 'Code_O', 'Code']

    df.drop(drop_names, 1, inplace=True)

def reduce_memory(df, df_name):
    
    # Memory usage before
    start_memory = df.memory_usage().sum() / 1024**2
    print('Memory usage of {} is {:.2f} MB'.format(df_name, start_memory))
    
    col_names = df.columns.values
    
    for col in col_names:
        col_type = df[col].dtype.name
        if col_type not in ['object', 'category', 'datetime64[ns, UTC]', 'bool']:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            
            else:
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_memory = df.memory_usage().sum() / 1024**2
    print('Memory usage for {} after optimization is: {:.2f} MB'.format(df_name, end_memory))
    print('Decreased by {:.1f}%'.format(100 * (start_memory - end_memory) / start_memory))



    
# ---------------------------------------------------------- Part 8
# Data preprocessing to reduce memory
# Data preprocessing to reduce memory
reduce_memory(news_train_df, 'news')
reduce_memory(market_train_df, 'market')
market_train_df.drop(['assetName', 'volume'], 1, inplace=True)

smalldata = False

if smalldata:
    market_train_df = market_train_df.tail(100_000)
    news_train_df = news_train_df.tail(300_000)

       
# simplify assetcode in news data
manipulate_news(news_train_df)
news_train_df['relev_sent'] = news_train_df.apply(lambda x: x['relevance'] * x['sentimentClass'], axis=1) 
news_train_df.drop(['relevance'], axis=1, inplace=True)




# ---------------------------------------------------------- Part 9
# Split date into before and after 22h (the time used in train data)
# E.g: 2007-03-07 23:26:39+00:00 -> 2007-03-08 00:00:00+00:00 (next day)
#      2009-02-25 21:00:50+00:00 -> 2009-02-25 00:00:00+00:00 (current day)
# Split date into before and after 22h (the time used in train data)
# E.g: 2007-03-07 23:26:39+00:00 -> 2007-03-08 00:00:00+00:00 (next day)
#      2009-02-25 21:00:50+00:00 -> 2009-02-25 00:00:00+00:00 (current day)
news_train_df['time'] = (news_train_df['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
# Round time of market_train_df to 0h of curret day
market_train_df['time'] = market_train_df['time'].dt.floor('1D')

news_cols_agg = {
    'urgency': ['max', 'sum'],
    'marketCommentary': ['max', 'sum'],
    'relev_sent': ['max', 'sum'],
    'sentimentNegative': ['max', 'sum'],
    'sentimentNeutral': ['max', 'sum'],
    'sentimentPositive': ['max', 'sum'],
    'noveltyCount12H': ['max', 'sum'],
    'noveltyCount24H': ['max', 'sum'],
    'noveltyCount3D': ['max', 'sum'],
    'noveltyCount5D': ['max', 'sum'],
    'noveltyCount7D': ['max', 'sum'],
    'volumeCounts12H': ['max', 'sum'],
    'volumeCounts24H': ['max', 'sum'],
    'volumeCounts3D': ['max', 'sum'],
    'volumeCounts5D': ['max', 'sum'],
    'volumeCounts7D': ['max', 'sum']}

# Aggregate numerical news features
news_train_df_aggregated = news_train_df.groupby(['time', 'assetCode']).agg(news_cols_agg)

# free memroy
del news_train_df
gc.collect()
reduce_memory(news_train_df_aggregated, 'news_train_df_aggregated')

# Flat columns
news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]


# Join with train
x = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])

# Free memory
del news_train_df_aggregated
del market_train_df
gc.collect()






# ---------------------------------------------------------- Part 10
# Set index ['assetCode', 'time'] for piece-wise exponentially weighted moving average based on different assets.
x.fillna(0, inplace=True)
# variables for exponential weighted moving average
variables = x.columns.values[14:26]
x.sort_values(['assetCode', 'time'], inplace=True)
x[variables] = x.groupby('assetCode').apply(lambda y: y[variables].ewm(span=7, min_periods=7).mean())

x.set_index(['assetCode', 'time'], inplace=True)

# Normalize the market data.
x['open'] = x.groupby('assetCode').apply(lambda y: (y['open']-y['open'].min())/
                                                 (y['open'].max()-y['open'].min())).values
x['close'] = x.groupby('assetCode').apply(lambda y: (y['close']-y['close'].min())/
                                                 (y['close'].max()-y['close'].min())).values
x.dropna(axis=0, how="any", inplace=True)

classification = True
try:
    if classification:
        y= x['returnsOpenNextMktres10']
        y[y<0] = 0
        y[y>=0] = 1
    else:
        y = x['returnsOpenNextMktres10']
    x.drop(columns=['returnsOpenNextMktres10'], inplace=True)
except:
    print("Train data generate wrong!")

try:
    universe = x['universe']
    time = x['time']
    x.drop(columns=['universe'], inplace=True)
except:
    print("'Universe' generate wrong!")

# adjust "relev_sent_sum" to [-1,1]
x['relev_sent_sigmoid'] = x.apply(lambda row: math.erf(row['relev_sent_sum']), axis=1)      

#??? don't know the meaning
# Fix some mixed-type columns
for bogus_col in ['marketCommentary_max', 'marketCommentary_sum']:
    x[bogus_col] = x[bogus_col].astype(float)
    
reduce_memory(x, 'x')





# ---------------------------------------------------------- Part 11
from sklearn.model_selection import train_test_split

X = x[['open', 'urgency_sum', 
       'marketCommentary_sum', 'relev_sent_sum', 'sentimentNegative_sum', 
       'sentimentNeutral_sum', 'sentimentPositive_sum', 'noveltyCount12H_sum', 
       'noveltyCount24H_sum', 'noveltyCount3D_sum', 'noveltyCount5D_sum',
       'noveltyCount7D_sum', 'volumeCounts12H_sum', 'volumeCounts24H_sum', 
       'volumeCounts3D_sum', 'volumeCounts5D_sum', 'volumeCounts7D_sum', 
       'relev_sent_sigmoid']]

X = x.copy()
Y = y.copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
