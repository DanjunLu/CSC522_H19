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

from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()

# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()
(market_train_df, news_train_df) = env.get_training_data()

import plotly.graph_objs as go
import numpy as np 
import pandas as pd
import re
import pandasql as ps
import datetime as dt
import gc
import math
import plotly.offline as py
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV    
from sklearn.svm import SVC 
from matplotlib import style
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from math import sqrt
from sklearn.metrics import mean_squared_error

style.use('ggplot')
np.random.seed(7)
py.init_notebook_mode(connected=True)



# Plot top 10 standard deviation with respect to open price change.
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
        title= 'Top 10 months of daily price change standard deviation',
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


# Top 10 recrds of price difference before replacement.
print(market_train_df.sort_values('price_diff').head(10)[['assetName', 'close', 'open']])

# Replace outliers in Open and Close price with their mean.
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

# Top 10 recrds of price difference after replacement.
print(market_train_df.sort_values('price_diff').head(10)[['assetName', 'close', 'open']])

top_std_plot(market_train_df, 5)

market_train_df.drop(['mean_close', 'mean_open', 'price_diff', 'CloseOpenRatio', 'universe'], axis=1, inplace=True)



# Stocks return quantiles plot.
data = []
for i in [0.05, 0.25, 0.5, 0.75, 0.95]:
    price_df = market_train_df.groupby('time')['returnsOpenNextMktres10'].quantile(i).reset_index()

    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['returnsOpenNextMktres10'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of open price return by quantiles",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = '10 Day Open Returns'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# Choose data after 2010-01-01
market_train_df = market_train_df.loc[market_train_df['time'] >= '2010-01-01 22:00:00+0000']
news_train_df = news_train_df.loc[news_train_df['time'] >= '2010-01-01 22:00:00+0000']


# News data seperate and expand by assetCodes with Regular Expression
def manipulate_news(df):
    
    # Using RE to get all assetCode with formats: "assetCode.O", "assetCode.N" and "assetCode"
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

    
# Reduce storage memory of data by optimizing the data type.
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
    
    

# Data preprocessing to reduce memory
reduce_memory(news_train_df, 'news')
reduce_memory(market_train_df, 'market')
market_train_df.drop(['assetName'], 1, inplace=True)

smalldata = False

if smalldata:
    market_train_df = market_train_df.tail(100_000)
    news_train_df = news_train_df.tail(300_000)

       
# simplify assetcode in news data
manipulate_news(news_train_df)

# Choose the stocks that we used in the model.
stocks = ['GOOG.O', 'YHOO.O', 'AAPL.O', 'MSFT.O', 'FB.O']
market_train_df = market_train_df.loc[market_train_df['assetCode'].isin(stocks)]
news_train_df = news_train_df.loc[news_train_df['assetCode'].isin(stocks)]


# Create new features.
news_train_df['relev_sent'] = news_train_df.apply(lambda x: x['relevance'] * x['sentimentClass'], axis=1) 
news_train_df.drop(['relevance'], axis=1, inplace=True)



# Split date into before and after 22h (the time used in train data)
# E.g: 2007-03-07 23:26:39+00:00 -> 2007-03-08 00:00:00+00:00 (next day)
#      2009-02-25 21:00:50+00:00 -> 2009-02-25 00:00:00+00:00 (current day)
news_train_df['time'] = (news_train_df['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
# Round time of market_train_df to 0h of curret day
market_train_df['time'] = market_train_df['time'].dt.floor('1D')


# Aggregate news data and join them into market data.
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


# Generate target variables.
try:
    temp = x['open']
    x['open_next_day'] = temp.groupby('assetCode').shift(-1)
    x.dropna(axis=0, inplace=True)
    y = x['open_next_day'] > x['open']
    x.drop(columns=['returnsOpenNextMktres10', 'open_next_day'], axis=1, inplace=True)
    
except:
    print("Train data generate wrong!")

# adjust "relev_sent_sum" to [-1,1]
x['relev_sent_sigmoid'] = x.apply(lambda row: math.erf(row['relev_sent_sum']), axis=1)      

reduce_memory(x, 'x')




# light gbm model for feature selection.
def lgb_model_func(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    params = dict(
        objective = 'binary',
        learning_rate = 0.1,
        num_leaves = 43,
        max_depth = -1,
        bagging_fraction = 0.75,
        bagging_freq = 2,
        feature_fraction = 0.5,
        lambda_l1 = 0.0,
        lambda_l2 = 1.0,
        metric = 'auc',
        seed = 42 # Change for better luck! :)
    )


    evals_result = {}
    train_data = lgb.Dataset(X_train, Y_train, feature_name=X_train.columns.tolist(),
                             free_raw_data=False)
    test_data = lgb.Dataset(X_test, Y_test, feature_name=X_train.columns.tolist(),
                             free_raw_data=False)

    lgb_model =  lgb.train(params, train_data, num_boost_round=3000, valid_sets=(test_data,), 
                           valid_names=('test',), verbose_eval=100,early_stopping_rounds=200, 
                           evals_result=evals_result)

    df_result = pd.DataFrame(evals_result['test'])
    
    return lgb_model
    
lgb_model = lgb_model_func(x, y)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
lgb.plot_importance(lgb_model, ax=ax[0])
lgb.plot_importance(lgb_model, ax=ax[1], importance_type='gain')
fig.tight_layout()





# Select features for modeling purpose based on Light GBM. 
features = ['open', 'close', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10','returnsClosePrevMktres10',
       'marketCommentary_sum', 'sentimentNeutral_max', 'relev_sent_max', 'sentimentNegative_max', 
        'sentimentNeutral_sum',  'sentimentPositive_max', 'sentimentPositive_sum']

X = x[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)




# K fold cross validation to choose logistic model and predict.
def k_fold_valid(X, y):
    
    k_fold = KFold(n_splits=10, shuffle=True)
    X = np.array(X)
    train_scores = []
    test_scores = []
    
    penalty = [0.01, 0.1, 1]
    
    for i in penalty:
        m = LogisticRegression(C = i, solver='newton-cg')
        train_score = []
        test_score = []
    
        for train_ind, test_ind in k_fold.split(X):
            X_train, X_test = X[train_ind], X[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]

            m.fit(X_train, y_train)

            train = m.score(X_train, y_train)
            test = m.score(X_test, y_test)

            train_score.append(train)
            test_score.append(test)
        
        train_scores.append(np.mean(train_score))
        test_scores.append(np.mean(test_score))
    
    ind = np.argmax(test_scores)
    
    return m, penalty[ind], train_scores[ind], test_scores[ind]

logistic_model, C, logistic_train_score, logistic_vali_score = k_fold_valid(X_train, y_train)
logistic_test_score = logistic_model.score(X_test, y_test)
print(logistic_train_score, logistic_test_score)






# create ANN model
def create_ANN_mdoel(penalty):
    
    model = Sequential()
    model.add(Dense(200, input_dim=len(features), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Cross validation to choose ANN model and predict.
def cross_valid(model_func, X, y):
    
    batch_sizes = [3000, 3500, 4000, 4500, 5000]
    
    valid_scores = []
    train_scores = []
    parameters = []
    
    for i in [0.01, 0.1, 1, 2, 3, 4]:
        m = model_func(i)
        for j in batch_sizes:
        
            history = m.fit(X, y, validation_split=0.1, epochs=200, batch_size=j, verbose=0)
            
            epoch = np.argmax(history.history['val_acc'])
            parameters.append((i, j))
            
            train_scores.append(history.history['acc'][-1])
            valid_scores.append(history.history['val_acc'][-1])
    
    ind = np.argmax(valid_scores)
    
    return parameters[ind][0], parameters[ind][1]

ANN_C, ANN_batch_sizes = cross_valid(create_ANN_mdoel, X_train, y_train)
ANN_model = create_ANN_mdoel(ANN_C)
history = ANN_model.fit(X_train, y_train, epochs=200, batch_size=ANN_batch_sizes, verbose=0)
ANN_train_score = ANN_model.evaluate(X_train, y_train, verbose=0)[1]
ANN_test_score = ANN_model.evaluate(X_test, y_test, verbose=0)[1]
print(ANN_train_score, ANN_test_score)


# Cross validation to choose SVM model and predict including Grid Search with C in (80,100) and gamma in (1,15).
def svm_cross_validation(X_train, y_train):    
       
    model = SVC(kernel='linear',max_iter=3000)    
    param_grid = {'C': range(80,100,2), 'gamma': range(1, 15,2)}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1,cv=10)    
    grid_search.fit(X_train, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])    
    model.fit(X_train, y_train)    
    
    return model

SVM_model = svm_cross_validation(X_train, y_train)

SVM_train_score = SVM_model.score(X_train, y_train)
SVM_test_score = SVM_model.score(X_test, y_test)
print(SVM_train_score, SVM_test_score)



#LSTM model
# data sets preparing
split1 = int(X.shape[0]*0.9)
split2 = int(split1*0.9)

x_0 = X.iloc[:split1,]
y_0 = y.iloc[:split1,]

x_train_LSTM = x_0.iloc[:split2,].values
y_train_LSTM = y_0.iloc[:split2,].values
x_valid_LSTM = x_0.iloc[split2:,].values
y_valid_LSTM = y_0.iloc[split2:,].values
x_test_LSTM = X.iloc[split1:,].values
y_test_LSTM = y.iloc[split1:,].values

# lstm need 3d input
x_train_LSTM = np.reshape(x_train_LSTM, (np.shape(x_train_LSTM)[0], 1, np.shape(x_train_LSTM)[1]) )
x_valid_LSTM = np.reshape(x_valid_LSTM, (np.shape(x_valid_LSTM)[0], 1, np.shape(x_valid_LSTM)[1]) )
x_test_LSTM = np.reshape(x_test_LSTM, (np.shape(x_test_LSTM)[0], 1, np.shape(x_test_LSTM)[1]) )



# build lstm model
def buildManyToOneModel(shape):
    
    model = Sequential()
    layers = [1, 50, 100, 1]  
    model.add(LSTM(layers[1], input_length=shape[1], input_dim=shape[2], kernel_regularizer = regularizers.l2(0.03), return_sequences = True))
    model.add(Dropout(0.1))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(layers[3]))
    model.add(Activation("sigmoid"))
    # output shape: (1, 1)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

LSTM_model = buildManyToOneModel(x_train_LSTM.shape)
callback = EarlyStopping(monitor="val_acc", patience=10, verbose=1, mode="auto")
history = LSTM_model.fit(x_train_LSTM, y_train_LSTM, epochs=512, batch_size=1024, validation_data = (x_valid_LSTM, y_valid_LSTM), callbacks=[callback], verbose = 0)


## plot result

# loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()


# accuracy
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.legend()
plt.show()

# rmse

yhat = LSTM_model.predict(x_test_LSTM)
rmse = sqrt(mean_squared_error(yhat, y_test_LSTM))
print('Test RMSE: %.3f' % rmse)

# evaluate result
LSTM_train_score = LSTM_model.evaluate(x_train_LSTM, y_train_LSTM, verbose=0)[1]
LSTM_test_score = LSTM_model.evaluate(x_test_LSTM, y_test_LSTM, verbose=0)[1]



# Aggregate the accuracy for all the models.
df = pd.DataFrame()
df['train_accuracy'] = [logistic_train_score, SVM_train_score, ANN_train_score, LSTM_train_score]
df['test_accuracy'] = [logistic_test_score, SVM_test_score, ANN_test_score, LSTM_test_score]
df['models'] = ['Logistic', 'SVM', 'ANN', 'LSTM']
print(df)



# Plot all the train and test accuracy.
fig, ax = plt.subplots()

n_groups = df.shape[0]
index = np.arange(n_groups)

bar_width = 0.35
opacity = 0.4

rects1 = ax.bar(index, df['train_accuracy'], bar_width,
                alpha=opacity, color='b',
                label='train_accuracy')

rects2 = ax.bar(index + bar_width, df['test_accuracy'], bar_width,
                alpha=opacity, color='r',
                label='test_accuracy')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Scores by models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df['models'])
ax.legend()

fig.tight_layout()
plt.ylim([0.7, 0.9])
plt.show()


#Reference
#Bruno G. do Amaral, A simple model - using the market and news data, https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data




