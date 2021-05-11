# Load all the necessary packages

# General utility
import datetime, time
import os
import gc
from copy import deepcopy

# Data manipulation
import numpy as np
import pandas as pd
import pickle

# Text processing
import re
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import itertools
import emoji

# Stock analysis
import ta # Technical Analysis library
import alpaca_trade_api as tradeapi # Alpaca trading api
from fredapi import Fred # FRED Api
import yfinance as yf # Yahoo Finance

# Plotting
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

# Neural Networks
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy

# Optimization
from scipy.optimize import minimize

# File paths
fpath = os.path.dirname(os.path.realpath(__file__))
invalid_ticker_f = fpath+'/invalid_tickers.txt'
contraction_f = fpath+'/contractions.txt'
slang_f = fpath+'/slang.txt'
ticker_f = fpath+'/tickers.csv'

#####################################################################################
#
# DATA CLEANING AND SENTIMENT ANALYSIS
#
#####################################################################################

# Self defined list of invalid ticker symbols
def loadInvalidTickers():
    with open(invalid_ticker_f,'r') as f:
        invalid_tickers = [line.strip() for line in f]

    return invalid_tickers

# Self defined list of contractions
def loadContractions():
    with open(contraction_f,'r') as f:
        contractions = {line.strip().split(':')[0]: line.strip().split(':')[1] for line in f}

    return contractions

# Emoticons
def loadEmoticons():
    
    return {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }

# Self defined list of slang terms
def loadSlang():
    with open(slang_f,'r') as f:
        slang = {line.strip().split(':')[0]: line.strip().split(':')[1] for line in f}

    return slang

# Load defined terms and set them to global variables
INVALID_TICKERS = loadInvalidTickers()
EMOTICONS = loadEmoticons()
CONTRACTIONS = loadContractions()
SLANG = loadSlang

def loadTickerPatterns():
    tickers = pd.read_csv(ticker_f)
    patterns = []
    for idx,row in tickers.iterrows():
        if row['Symbol'] not in INVALID_TICKERS:
            pattern = r'\b'+row['Symbol']+r'\b'
            prog = re.compile(pattern)
            patterns.append([prog,row['Symbol']])

    fullnames = [['GAMESTOP','GME'],['TESLA','TSLA'],['APPLE','AAPL'],['FORD','F'],['AMAZON','AMZN'],['ARKK','ARKK'],['VIX','VIX'],['PALANTIR','PLTR'],['BLACKBERRY','BB'],['GROWGENERATION','GRWG'],['ALTRIA','MO'],['SUNDIAL','SNDL'],['VIACOM','VIAC']]
    for name in fullnames:
        pattern = r'\b'+name[0]+r'\b'
        prog = re.compile(pattern)
        patterns.append([prog,name[1]])
    return patterns

TICKERS = loadTickerPatterns()

def extractTickrs(comment):
    found_tickrs = []
    for pattern in TICKERS:
        if pattern[0].search(comment) != None:
            comment = re.sub(pattern[0],'',comment)
            found_tickrs.append(pattern[1])
    return comment, list(set(found_tickrs))

def clean_comment(comment):    
    
    #Escaping HTML characters
    #comment = BeautifulSoup(comment).get_text()
   
    #Special case not handled previously.
    comment = comment.replace('\x92',"'")
    
    #Removal of hastags/account/subreddit/address/tags
    comment = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|(\br/.*\b)|(\w+:\/\/\S+)|(<[^>]+>)", " ", comment).split())
    
    #Remove quotes
    comment = comment.replace('"','')
    
    #Lower case
    comment = comment.lower()
    
    #CONTRACTIONS source: https://en.wikipedia.org/wiki/Contraction_%28grammar%29
    comment = comment.replace("’","'")
    words = comment.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    comment = " ".join(reformed)
    
    #Deal with emoticons source: https://en.wikipedia.org/wiki/List_of_emoticons 
    words = comment.split()
    reformed = [EMOTICONS[word] if word in EMOTICONS else word for word in words]
    comment = " ".join(reformed)
    
    #Deal with emojis
    comment = emoji.demojize(comment)
    
    # Remove punctuations and numbers
    comment = re.sub('[^a-zA-Z_]', ' ', comment)

    #Remove tickrs
    comment, ticks = extractTickrs(comment.upper())
    
    # Standardizing words
    comment = ''.join(''.join(s)[:2] for _, s in itertools.groupby(comment))
    
    # Single character removal
    comment = re.sub(r"\s+[a-zA-Z]\s+", ' ', comment)

    # Removing multiple spaces
    comment = re.sub(r'\s+', ' ', comment)

    return comment.lower().strip(),ticks

def getSentiments(comments):
    all_sentiment = []
    tickr_sentiment = []
    processed_comments = []

    for comment in comments:
        # Skip over deleted or removed comments
        if comment.body == '[deleted]' or comment.body == '[removed]':
            pass

        else:
            # Clean the comment and retrive ticker symbols
            cleaned,ticks = clean_comment(comment.body)

            # Get sentiment score
            ss = sid.polarity_scores(cleaned)
            if ss['compound'] > 0.05:
                neg = 0
                neu = 0
                pos = 1
            elif ss['compound'] < -0.5:
                neg = 1
                neu = 0
                pos = 0
            else:
                neg = 0
                neu = 1
                pos = 0

            processed_comments.append([comment.body,';'.join(ticks),comment.score,ss['neg'],
                                  ss['neu'],
                                  ss['pos'],
                                  ss['compound'],
                                  neg, 
                                  neu, 
                                  pos])

            all_sentiment.append([comment.created_utc, 
                                  comment.score, 
                                  ss['neg'],
                                  ss['neu'],
                                  ss['pos'],
                                  ss['compound'],
                                  neg, 
                                  neu, 
                                  pos])

            for tick in ticks:
                tickr_sentiment.append([tick,
                                        comment.created_utc,
                                  comment.score, 
                                  ss['neg'],
                                  ss['neu'],
                                  ss['pos'],
                                  ss['compound'],
                                  neg, 
                                  neu, 
                                  pos])

    processed_comments = pd.DataFrame(processed_comments, 
                                 columns=['Comment',
                                        'Stocks Mentioned', 
                                          'Comment Score',
                                          'Neg_Score',
                                          'Neu_Score',
                                          'Pos_Score',
                                          'Compound_Score',
                                          'IsNeg',
                                          'IsNeu',
                                          'IsPos'])
    all_sentiment = pd.DataFrame(all_sentiment, 
                                 columns=['Created_UTC', 
                                          'Score',
                                          'Neg_Score',
                                          'Neu_Score',
                                          'Pos_Score',
                                          'Compound_Score',
                                          'IsNeg',
                                          'IsNeu',
                                          'IsPos'])
    tickr_sentiment = pd.DataFrame(tickr_sentiment, 
                                   columns=['Tickr',
                                            'Created_UTC', 
                                          'Score',
                                          'Neg_Score',
                                          'Neu_Score',
                                          'Pos_Score',
                                          'Compound_Score',
                                          'IsNeg',
                                          'IsNeu',
                                          'IsPos'])
    
    return all_sentiment, tickr_sentiment, processed_comments

def getSentimentSummary(df,date):
    # Add 1 to score to count the author
    scores = df['Score'] + 1
    wscores = scores.abs().sum()

    # Total comments
    n_comments = len(df)
    
    # Compute average sentiment scores
    avg_score = scores.mean()
    avg_neg_score = df['Neg_Score'].mean()
    avg_neu_score = df['Neu_Score'].mean()
    avg_pos_score = df['Pos_Score'].mean()
    avg_compound_score = df['Compound_Score'].mean()

    # Tally number of neg, neu, and pos sentiment comments
    n_neg = df['IsNeg'].sum()
    n_neu = df['IsNeu'].sum()
    n_pos = df['IsPos'].sum()

    # Tally weighted number of neg, neu, and pos sentiment comments
    w_n_neg = (df['IsNeg'] * scores).sum()
    w_n_neu = (df['IsNeu'] * scores).sum()
    w_n_pos = (df['IsPos'] * scores).sum()

    # Percentage sentiment
    p_neg = n_neg / n_comments
    p_neu = n_neu / n_comments
    p_pos = n_pos / n_comments

    # Computed weighted averages
    if wscores == 0:
        # if sum of scores is 0 assume everything is neutral
        # Compute weighted average sentiment scores based on comment score
        w_avg_neg_score = 0
        w_avg_neu_score = 1
        w_avg_pos_score = 0
        w_avg_compound_score = 0

        # Weighted Percentage sentiment
        w_p_neg = 0
        w_p_neu = 1
        w_p_pos = 0

    else:
        # Compute weighted average sentiment scores based on comment score
        w_avg_neg_score = (df['Neg_Score'] * scores).sum() / wscores
        w_avg_neu_score = (df['Neu_Score'] * scores).sum() / wscores
        w_avg_pos_score = (df['Pos_Score'] * scores).sum() / wscores
        w_avg_compound_score = (df['Compound_Score'] * scores).sum() / wscores

        # Weighted Percentage sentiment
        w_p_neg = w_n_neg / wscores
        w_p_neu = w_n_neu / wscores
        w_p_pos = w_n_pos / wscores

    return [date,
        n_comments,
        avg_score,
        avg_neg_score,
        avg_neu_score,
        avg_pos_score,
        avg_compound_score,
        w_avg_neg_score,
        w_avg_neu_score,
        w_avg_pos_score,
        w_avg_compound_score,
        n_neg,
        n_neu,
        n_pos,
        w_n_neg,
        w_n_neu,
        w_n_pos,
        p_neg,
        p_neu,
        p_pos,
        w_p_neg,
        w_p_neu,
        w_p_pos
    ]

def processComments(file):
    # Get the date of comments
    split = file[:-4].split('_')
    date = '-'.join([split[-3],split[-2],split[-1]])
    
    # Load the raw comments
    with open(file,'rb') as f:
        comments = pickle.load(f)

    # Extract sentiment
    all_sentiment, tickr_sentiment,processed_comments = getSentiments(comments)
    
    # Summarize total sentiment
    sumry = getSentimentSummary(all_sentiment,date)
    
    # Summarize sentiment for each ticker
    tsum = []
    for t in tickr_sentiment['Tickr'].unique():
        tsumry = getSentimentSummary(tickr_sentiment[tickr_sentiment['Tickr']==t],date)
        tsumry.insert(0, t)
        tsum.append(tsumry)

    # Return results
    return len(comments), sumry, tsum, processed_comments

def plotWordCloud(df, width=3000, height=2000, figsize=(40,30), stopwords=None, background_color = 'black',cmap='RdYlGn',filename=None):
    wordlist = dict(zip(df['Tickr'],df['#Comments']))
    norm = matplotlib.colors.SymLogNorm(linthresh=0.05,linscale=0.3,vmin=-1,vmax=1)
    CMap = plt.get_cmap(cmap)
    word_colors = dict(zip(df['Tickr'],CMap(norm(df['Avg_Compound_Score']))))
    def color_func(word, *args, **kwargs):
        return matplotlib.colors.rgb2hex(word_colors[word])
    wordcloud = WordCloud(
            width = width,
            height = height,
            background_color = background_color,
            stopwords = stopwords,
            collocations=False,
            color_func=color_func).generate_from_frequencies(wordlist)
    fig = plt.figure(
        figsize = figsize,
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    


#####################################################################################
#
# DATA LOADING
#
#####################################################################################
# dataloader class to help import data
class MyDataLoader:
    def __init__(self, live=False,alpaca_ID=None, alpaca_key=None,fred_key=None,today_all=None,today_tickr=None, tomorrow_all=None,tomorrow_tickr=None):
        # Connection Info
        self.alpaca_apiID = alpaca_ID
        self.alpaca_apiKey = alpaca_key
        if live:
            self.alpaca_base_url = 'https://api.alpaca.markets'
        else:
            self.alpaca_base_url = 'https://paper-api.alpaca.markets'
        self.alpaca_data_url = 'https://data.alpaca.markets'
        self.fred_apiKey = fred_key
        self.today_all = today_all
        self.today_tickr = today_tickr
        self.tomorrow_all = tomorrow_all
        self.tomorrow_tickr = tomorrow_tickr
        self.scalar = None
        
        # Setup Alpaca API Connection
        if self.alpaca_apiID is not None:
            self.api = tradeapi.REST(
                self.alpaca_apiID,
                self.alpaca_apiKey,
                self.alpaca_base_url
            )
        
        # Setup FRED API Connection
        if self.fred_apiKey is not None:
            self.fred = Fred(api_key = self.fred_apiKey)
        
    def getAlpacaData(self, ticker, end, adjustment='raw'):
        # Load data from Alpaca
        data = self.api.get_bars(ticker, tradeapi.rest.TimeFrame.Minute, end, end,adjustment='raw')
        
        # Get open price
        open_price = float(data._raw[-1]['c'])
        
        # Return open price
        return open_price
    
    def getYahooData(self, ticker, start, end):
        # Load data from yahoo
        df = yf.download(ticker, 
                      start=start, 
                      end=end, 
                      progress=False)
        
        return df
    
    def getFredData(self,start='2000-01-01'):
        # Get these FRED series
        fredseries = ['DGS1','DGS5','DGS30','DGS2','DGS3MO','DGS1MO','DGS10',
                      'T10YIE','DAAA','CCSA','ICSA','MORTGAGE15US','POILWTIUSDM',
                      'TOTALSA','CCSA','DAUPSA','CPALTT01USM657N','STLFSI2','UMCSENT']
        
        # Load all series data into a single dataframe
        df = pd.DataFrame(self.fred.get_series(fredseries[0],observation_start=start),columns=[fredseries[0]])
        for i in range(1,len(fredseries)):
            df2 = pd.DataFrame(self.fred.get_series(fredseries[i]),columns=[fredseries[i]])
            df = df.merge(df2,how='outer',left_index=True,right_index=True)
        
        # Interpolate missing values
        df = df.interpolate()
        
        # Convert to percent change
        df = df.pct_change()
        
        # Return DataFrame  
        return df
    
    def getTAData(self, df):
        ## Technical Indicators
        # Adding all the indicators
        augdf = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        
        # Dropping everything else besides 'Close' and the Indicators
        augdf.drop(['Close','High', 'Low', 'Volume','Adj Close'], axis=1, inplace=True)
        
        # Return DataFrame
        return augdf
    
    def getTAData2(self, df):
        ## Technical Indicators
        # Adding all the indicators
        augdf = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        
        # Dropping everything else besides 'Close' and the Indicators
        augdf.drop(['High', 'Low', 'Volume','Adj Close'], axis=1, inplace=True)
        
        # Return DataFrame
        return augdf

    def getComments(self,tickr):
        # load commments
        with open(self.today_all,'rb') as f:
            today_all = pickle.load(f)
        with open(self.today_tickr,'rb') as f:
            today_tickr = pickle.load(f)
        with open(self.tomorrow_all,'rb') as f:
            tomorrow_all = pickle.load(f)
        with open(self.tomorrow_tickr,'rb') as f:
            tomorrow_tickr = pickle.load(f)
        
        # Filter to only the relevant tickr
        today_tickr = today_tickr[today_tickr['Tickr']==tickr]
        tomorrow_tickr = tomorrow_tickr[tomorrow_tickr['Tickr']==tickr]
        today_tickr.drop(['Tickr'],axis=1,inplace=True)
        tomorrow_tickr.drop(['Tickr'],axis=1,inplace=True)
        
        # Rename columns before merging
        today_all_cols = {item: 'Today_All_'+item for item in today_all.columns}
        today_tickr_cols = {item: 'Today_Tickr_'+item for item in today_tickr.columns}
        tomorrow_all_cols = {item: 'Tomorrow_All_'+item for item in tomorrow_all.columns}
        tomorrow_tickr_cols = {item: 'Tomorrow_Tickr_'+item for item in tomorrow_tickr.columns}
        today_all_cols['Date'] = 'Date'
        today_tickr_cols['Date'] = 'Date'
        tomorrow_all_cols['Date'] = 'Date'
        tomorrow_tickr_cols['Date'] = 'Date'
        today_all.rename(columns=today_all_cols,inplace=True)
        today_tickr.rename(columns=today_tickr_cols,inplace=True)
        tomorrow_all.rename(columns=tomorrow_all_cols,inplace=True)
        tomorrow_tickr.rename(columns=tomorrow_tickr_cols,inplace=True)
        
        # Merge the dataframes
        df = today_all.merge(today_tickr,on='Date',how='outer').merge(tomorrow_all,on='Date',how='outer').merge(tomorrow_tickr,on='Date',how='outer')
        
        # Create date index
        df.index = pd.to_datetime(df['Date'])
        df.drop(['Date'],axis=1,inplace=True)
        
        return df
    
    def loadData(self, ticker, start, end):
        
        # Get stock data
        #df = self.getAlpacaData(ticker, start, end)
        df = self.getYahooData(ticker, '2000-01-01', end)
        
        # Add ta data
        df = self.getTAData(df)
        
        # Get comment data
        c = self.getComments(ticker)
        
        # Merge data
        df = df.merge(c,how='left',left_index=True,right_index=True)
        df.fillna(0,inplace=True)
        
        # Calculate the difference
        df['Change'] = df['Open'].diff()
        
        # Calculate the pct change
        df['P_Change'] = df['Open'].pct_change().shift(-1)
        
        # Shift the tomorrow comments up
        tomorrow_cols = [col for col in df.columns if col[0:8]=='Tomorrow']
        for col in tomorrow_cols:
            df[col] = df[col].shift(-1)
        
        # Trim dataseries
        df = df[start:end]
        
        return df
    
    def loadDataOld(self, ticker, start, end, freddata):
        
        # Get stock data
        #df = self.getAlpacaData(ticker, start, end)
        df = self.getYahooData(ticker, '2000-01-01', end)

        # Merge fred data
        df = df.merge(freddata,how='left',left_index=True,right_index=True)
        
        # Interpolate missing values
        df = df.interpolate()
        
        # Add ta data
        df = self.getTAData(df)
        
        # Get comment data
        c = self.getComments(ticker)
        
        # Merge data
        df = df.merge(c,how='left',left_index=True,right_index=True)
        df.fillna(0,inplace=True)
        
        # Calculate the difference
        df['Change'] = df['Open'].diff()
        
        # Calculate the pct change
        df['P_Change'] = df['Open'].pct_change().shift(-1)
        
        # Shift the tomorrow comments up
        tomorrow_cols = [col for col in df.columns if col[0:8]=='Tomorrow']
        for col in tomorrow_cols:
            df[col] = df[col].shift(-1)
        
        # Trim dataseries
        df = df[start:end]
        
        # Dropping any NaNs
        #df.dropna(inplace=True)
        
        return df

    def loadClassificationData(self, ticker, start, end):
        
        # Get stock data
        #df = self.getAlpacaData(ticker, start, end)
        df = self.getYahooData(ticker, '2000-01-01', end)
        
        # Add ta data
        df = self.getTAData2(df)
        
        # Get comment data
        c = self.getComments(ticker)
        
        # Merge data
        df = df.merge(c,how='left',left_index=True,right_index=True)
        df.fillna(0,inplace=True)
        
        # Calculate the difference
        df['Change'] = df['Close'] - df['Open']
        
        # Calculate the pct change
        df['P_Change'] = df['Change'] / df['Open']    

        # Today's open
        df['Today_Open'] = df['Open'].shift(-1)

        # After hours change
        df['After_hours_change'] = df['Today_Open'] - df['Close']
        df['After_hours_pchange'] = df['After_hours_change'] / df['Close']

        # Calculate buy flag
        df['Buy'] = df['Change'].apply(lambda x: 1 if x >= 0 else 0)

        # Shift the tomorrow comments up
        tomorrow_cols = [col for col in df.columns if col[0:8]=='Tomorrow']
        for col in tomorrow_cols:
            df[col] = df[col].shift(-1)
        
        # Trim dataseries
        df = df[start:end]

        # move the buy column to the front
        col = df.pop("Buy")
        df.insert(0, col.name, col)
        
        return df

    def loadPredictionData(self, ticker, start, end):
        
        # Get stock data
        #df = self.getAlpacaData(ticker, start, end)
        df = self.getYahooData(ticker, '2000-01-01', end)

        # Add ta data
        df = self.getTAData(df)
        
        # Get comment data
        c = self.getComments(ticker)

        # Add in today's opening price
        open_price = self.getAlpacaData(ticker, end)
        df.loc[pd.to_datetime(end),'Open'] = open_price
        
        # Merge data
        df = df.merge(c,how='left',left_index=True,right_index=True)
        df.fillna(0,inplace=True)

        # Calculate the difference
        df['Change'] = df['Open'].diff()
        
        # Calculate the pct change
        df['P_Change'] = df['Open'].pct_change().shift(-1)
        
        # Shift the tomorrow comments up
        tomorrow_cols = [col for col in df.columns if col[0:8]=='Tomorrow']
        for col in tomorrow_cols:
            df[col] = df[col].shift(-1)
        
        # Trim dataseries
        df = df[start:end]
        
        # Dropping any NaNs
        #df.dropna(inplace=True)
        
        return df

    def loadPredictionDataOld(self, ticker, start, end, freddata):
        
        # Get stock data
        #df = self.getAlpacaData(ticker, start, end)
        df = self.getYahooData(ticker, '2000-01-01', end)
        
        # Add in today's opening price
        open_price = self.getAlpacaData(ticker, end)
        df.loc[pd.to_datetime(end),'Open'] = open_price

        # Merge fred data
        df = df.merge(freddata,how='left',left_index=True,right_index=True)
        
        # Interpolate missing values
        df = df.interpolate()
        
        # Add ta data
        df = self.getTAData(df)
        
        # Get comment data
        c = self.getComments(ticker)
        
        # Merge data
        df = df.merge(c,how='left',left_index=True,right_index=True)
        df.fillna(0,inplace=True)
        
        # Calculate the difference
        df['Change'] = df['Open'].diff()
        
        # Calculate the pct change
        df['P_Change'] = df['Open'].pct_change().shift(-1)
        
        # Shift the tomorrow comments up
        tomorrow_cols = [col for col in df.columns if col[0:8]=='Tomorrow']
        for col in tomorrow_cols:
            df[col] = df[col].shift(-1)
        
        # Trim dataseries
        df = df[start:end]
        
        # Dropping any NaNs
        #df.dropna(inplace=True)
        
        return df

    def train_test_split(self, data, n_steps_in, n_steps_out,n_test):
        """
        Splits the multivariate time sequence
        """
        # Creating a list for storing transformed data
        X, y = [], []

        # Create training data
        for i in range(len(data)-n_steps_in):
            # Finding the end of the current sequence
            end = i + n_steps_in
            out_end = end + n_steps_out

            # Breaking out of the loop if we have exceeded the dataset's length
            if out_end > len(data):
                break

            # Splitting the sequences into: x = past prices and indicators, y = prices ahead
            seq_x, seq_y = data[i:end, :], data[end:out_end, 0]

            X.append(seq_x)
            y.append(seq_y)
        
        return Variable(torch.Tensor(np.array(X[:-n_test]))), Variable(torch.Tensor(np.array(y[:-n_test]))),Variable(torch.Tensor(np.array(X[-n_test:]))),Variable(torch.Tensor(np.array(y[-n_test:])))
    
    def sequinize(self, data, n_steps_in, n_steps_out):
        """
        Splits the multivariate time sequence
        """
        # Creating a list for storing transformed data
        X, y = [], []

        # Create training data
        for i in range(len(data)-n_steps_in):
            # Finding the end of the current sequence
            end = i + n_steps_in
            out_end = end + n_steps_out

            # Breaking out of the loop if we have exceeded the dataset's length
            if out_end > len(data):
                break

            # Splitting the sequences into: x = past prices and indicators, y = prices ahead
            seq_x, seq_y = data[i:end, :], data[end:out_end, 0]

            X.append(seq_x)
            y.append(seq_y)
        
        return Variable(torch.Tensor(np.array(X[:-n_test]))), Variable(torch.Tensor(np.array(y[:-n_test]))),Variable(torch.Tensor(np.array(X[-n_test:]))),Variable(torch.Tensor(np.array(y[-n_test:])))
          

#####################################################################################
#
# NEURAL NETWORKS
#
#####################################################################################        

# Pytorch - for easy loading of data
class StockDataset(Dataset):
    def __init__(self,x, y):
        self.x = x
        self.y = y
        self.length = len(y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


# https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch
class StockLSTM(nn.Module):
    
    def __init__(self,tickr,n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,TRAIN_ON_GPU=False,TRAIN_ON_MULTI_GPUS=False):
        '''
        Initialize LSTM Model for predicting stock prices.
        
        Arguments:
            n_features: Number of features used for prediction
            n_periods: Number of time periods used for prediction
            n_hidden: Number of neurons to use in a hidden layer
            n_layers: Number of hidden layers in the LSTM
            drop_prob: Dropout rate
            lr: learning rate
        '''
        # Call super method
        super().__init__()
        
        # Save configurations
        self.name = tickr
        self.n_features = n_features
        self.n_periods = n_periods
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.clip = clip
        self.TRAIN_ON_GPU = TRAIN_ON_GPU
        self.TRAIN_ON_MULTI_GPUS = TRAIN_ON_MULTI_GPUS
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size = n_features, 
                                 hidden_size = n_hidden,
                                 num_layers = n_layers, 
                                 dropout=drop_prob,
                                 batch_first = True)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Define the fully connected layer
        self.fc = nn.Linear(n_hidden*n_periods, 1)
        
    def init_hidden(self, batch_size):
        '''
        Initialize hidden state.
        Create two new tensors with sizes n_layers x batch_size x n_hidden,
        initialized to zero, for hidden state and cell state of LSTM
        Arguments:
            batch_size: batch size, an integer
        Returns:
            hidden: hidden state initialized
        '''
        weight = next(self.parameters()).data
        if self.TRAIN_ON_MULTI_GPUS or self.TRAIN_ON_GPU:
            hidden = (weight.new(batch_size, self.n_layers, self.n_hidden).zero_().cuda(),
                      weight.new(batch_size, self.n_layers, self.n_hidden).zero_().cuda())

        else:
            hidden = (weight.new(batch_size, self.n_layers, self.n_hidden).zero_(),
                      weight.new(batch_size, self.n_layers, self.n_hidden).zero_())

        return hidden
    
    def forward(self, x, hidden):  
        '''
        Forward pass through the network.
        These inputs are x and the hidden/cell state.
        
        Arguments:
            x: Shaped (seq_len, batch, input_size)
            hidden: Shaped (num_layers * num_directions, batch, hidden_size)
        
        Returns:
            out: Shaped (seq_len, batch, num_directions * hidden_size)
            hidden: Shaped (num_layers * num_directions, batch, hidden_size)
        '''
        # Get batch size
        bs = x.size()[0]
        
        # Reshape hidden state
        hidden = tuple([h.permute(1,0,2).contiguous() for h in hidden])
        
        # Get LSTM output
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Add dropoutlayer
        out = self.dropout(lstm_out)
        
        # Shape the output to be (batch_size * n_periods, hidden_dim)
        out = out.contiguous().view(bs,-1)

        # Get final prediction
        out = self.fc(out)
        
        # Return final output and the hidden state
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])
        return out, hidden
    
    def predict(self, inputs, hidden):
        model.eval()
        
        # Get batch size
        batch_size = inputs.size()[0]

        # Evaluate on data
        try:
            val_h = model.init_hidden(batch_size)
        except AttributeError:
            # if using DataParallel wrapper to use multipl GPUs
            val_h = model.module.init_hidden(batch_size)
                
        if self.TRAIN_ON_GPU or self.TRAIN_ON_MULTI_GPUS:
            inputs = inputs.cuda()

        # get output of model
        output, hidden = model(inputs, val_h)

        return output, hidden

class StockLSTMClassifier(nn.Module):
    
    def __init__(self,tickr,n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,TRAIN_ON_GPU=False,TRAIN_ON_MULTI_GPUS=False):
        '''
        Initialize LSTM Model for predicting stock prices.
        
        Arguments:
            n_features: Number of features used for prediction
            n_periods: Number of time periods used for prediction
            n_hidden: Number of neurons to use in a hidden layer
            n_layers: Number of hidden layers in the LSTM
            drop_prob: Dropout rate
            lr: learning rate
        '''
        # Call super method
        super().__init__()
        
        # Save configurations
        self.name = tickr
        self.n_features = n_features
        self.n_periods = n_periods
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.clip = clip
        self.TRAIN_ON_GPU = TRAIN_ON_GPU
        self.TRAIN_ON_MULTI_GPUS = TRAIN_ON_MULTI_GPUS
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size = n_features, 
                                 hidden_size = n_hidden,
                                 num_layers = n_layers, 
                                 dropout=drop_prob,
                                 batch_first = True)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Define the fully connected layer
        self.fc = nn.Linear(n_hidden*n_periods, 1)
        
    def init_hidden(self, batch_size):
        '''
        Initialize hidden state.
        Create two new tensors with sizes n_layers x batch_size x n_hidden,
        initialized to zero, for hidden state and cell state of LSTM
        Arguments:
            batch_size: batch size, an integer
        Returns:
            hidden: hidden state initialized
        '''
        weight = next(self.parameters()).data
        if self.TRAIN_ON_MULTI_GPUS or self.TRAIN_ON_GPU:
            hidden = (weight.new(batch_size, self.n_layers, self.n_hidden).zero_().cuda(),
                      weight.new(batch_size, self.n_layers, self.n_hidden).zero_().cuda())

        else:
            hidden = (weight.new(batch_size, self.n_layers, self.n_hidden).zero_(),
                      weight.new(batch_size, self.n_layers, self.n_hidden).zero_())

        return hidden
    
    def forward(self, x, hidden):  
        '''
        Forward pass through the network.
        These inputs are x and the hidden/cell state.
        
        Arguments:
            x: Shaped (seq_len, batch, input_size)
            hidden: Shaped (num_layers * num_directions, batch, hidden_size)
        
        Returns:
            out: Shaped (seq_len, batch, num_directions * hidden_size)
            hidden: Shaped (num_layers * num_directions, batch, hidden_size)
        '''
        # Get batch size
        bs = x.size()[0]
        
        # Reshape hidden state
        hidden = tuple([h.permute(1,0,2).contiguous() for h in hidden])
        
        # Get LSTM output
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Add dropoutlayer
        out = self.dropout(lstm_out)
        
        # Shape the output to be (batch_size * n_periods, hidden_dim)
        out = out.contiguous().view(bs,-1)

        # Get final prediction
        out = self.fc(out)
        #out = torch.squeeze(out, 1)
        out = torch.sigmoid(out)
        
        # Return final output and the hidden state
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])
        return out, hidden

# train the model!
def train(model, dataloaders, criterion, optimizer, epochs=10, log_level='Quiet',save=False,plot=False,TRAIN_ON_GPU=False,TRAIN_ON_MULTI_GPUS=False):
    '''
    Training a network and serialize it to local file.

    Arguments:    
        model: StockLSTM model
        dataloaders: a dictionary of train, validation, and test dataloaders
        epochs: Number of epochs to train
        lr: learning rate
        clip: gradient clipping
    '''
    since = time.time()
    
    # Put the model in training mode
    model.train()

    # define optimizer and loss
    opt = optimizer
    criterion = criterion


    if TRAIN_ON_MULTI_GPUS:
        #print('Training on multiple GPUs')
        model = nn.DataParallel(model).cuda()
    elif TRAIN_ON_GPU:
        #print('Training on single GPU')
        model = model.cuda()
    else:
        print('Training on CPU...')

    # list to contain losses to be plotted
    losses = []
    vlosses = []

    counter = 0

    for e in range(epochs):
        
        train_losses = []

        for inputs, targets in dataloaders['train']:
            # Get batch size
            batch_size = inputs.size()[0]
            
            # initialize hidden state
            try:
                hidden = model.init_hidden(batch_size)
            except:
                # if using DataParallel wrapper to use multiple GPUs
                hidden = model.module.init_hidden(batch_size)
            
            # If using a GPU, send the data to the device
            if TRAIN_ON_GPU or TRAIN_ON_MULTI_GPUS:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise we'd backprop
            # through the entire training history.
            hidden = tuple([each.data for each in hidden])

            # zero acculated gradients
            model.zero_grad()
            
            with torch.set_grad_enabled(True):
                # get output from model
                output, hidden = model(inputs, hidden)

                # log
                if log_level=='debug':
                    print(output.size())
                # calculate the loss and perform backprop
                loss = criterion(output, targets)
                loss.backward()
                train_losses.append(loss.item())

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
                try:
                    nn.utils.clip_grad_norm_(model.parameters(), model.clip)
                except:
                    # if using DataParallel wrapper to use multiple GPUs
                    nn.utils.clip_grad_norm_(model.parameters(), model.module.clip)

                opt.step()

        val_losses = []
        model.eval()
        for inputs, targets in dataloaders['test']:
            # Get batch size
            batch_size = inputs.size()[0]
            
            # Evaluate on test data
            try:
                val_h = model.init_hidden(batch_size)
            except AttributeError:
                # if using DataParallel wrapper to use multipl GPUs
                val_h = model.module.init_hidden(batch_size)
            
            # if using GPUs, load test data to device
            if TRAIN_ON_GPU or TRAIN_ON_MULTI_GPUS:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(False):
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output, targets)
                val_losses.append(val_loss.item())

        # append loss into losses list
        losses.append(np.mean(train_losses))
        vlosses.append(np.mean(val_losses))


        # Put model back in training mode
        model.train()
        if log_level in ['detailed','debug']:
            print(f'Epoch: {e+1}/{epochs}...',
                      f'Loss: {np.mean(train_losses):.4f}...',
                      f'Val Loss: {np.mean(val_losses):.4f}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    if save:
        with open(savedmodels+model_name, 'wb') as f:
            torch.save(model, f)

    # plot loss curve
    if plot:
        loss_plot(losses, vlosses)

    # Get final losses
    m_train_loss = np.mean(losses,dtype=np.float64)
    m_test_loss = np.mean(val_losses,dtype=np.float64)

    # Clean up memory
    del since
    del time_elapsed
    del inputs
    del output
    del targets
    del losses
    del val_losses
    del val_loss
    gc.collect()
    torch.cuda.empty_cache()

    return m_train_loss, m_test_loss

# Utility to plot learning curve
def loss_plot(losses, valid_losses):
    '''
    Plot the validation and training loss.

    Arguments:
        losses: A list of training losses
        valid_losses: A list of validation losses
    Returns:
        No returns, just plot the graph.
    '''
    # losses and valid_losses should have same length
    assert len(losses) == len(valid_losses)
    epochs = np.arange(len(losses))
    plt.plot(epochs, losses, 'r-', valid_losses, 'b-')        

def setupPytorch():
    TRAIN_ON_GPU = torch.cuda.is_available()
    TRAIN_ON_MULTI_GPUS = (torch.cuda.device_count() >= 2)
    gpus = torch.cuda.device_count()

    if TRAIN_ON_MULTI_GPUS:
        print(f"Training on {gpus} GPUs!")

    elif TRAIN_ON_GPU:
        print('Training on GPU!')

    else: 
        print('No GPU available, training on CPU; consider making n_epochs very small.')  

    return TRAIN_ON_GPU, TRAIN_ON_MULTI_GPUS
          
def gridSearch(grid_results,tickr,n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,batch_size,data, epochs, n_test,n_steps_out,fpath,TRAIN_ON_GPU=False,TRAIN_ON_MULTI_GPUS=False):
    
    # Keep track of time
    grid_since = time.time()

    # Create list of all combinations to search
    configurations = np.array(np.meshgrid(n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,batch_size)).T.reshape(-1,8)
    print(f'Hyper Parameter search across {len(configurations)} different configurations.')

    # Get a data loader
    dl = MyDataLoader()

    # For each configuration, build a model and train
    coun = 0
    for config in configurations:
        coun +=1
        print(f'Iteration: {coun} |periods: {int(config[1])}|hidden: {int(config[2])}|layers: {int(config[3])}|drop: {config[4]}|lr: {config[5]}|clip: {int(config[6])}')
        # Build model
        model = StockLSTM(tickr=tickr,
            n_features=int(config[0]),
            n_periods=int(config[1]), 
            n_hidden=int(config[2]), 
            n_layers=int(config[3]), 
            drop_prob=config[4], 
            lr=config[5],
            clip=int(config[6]),
            TRAIN_ON_GPU=TRAIN_ON_GPU,
            TRAIN_ON_MULTI_GPUS=TRAIN_ON_MULTI_GPUS)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config[5])

        # Build dataset
        X_train, y_train, X_test, y_test = dl.train_test_split(data=data.to_numpy(), n_steps_in=int(config[1]), n_steps_out=n_steps_out,n_test=n_test)
        stock_datasets = {}
        stock_datasets['train'] = StockDataset(X_train,y_train)
        stock_datasets['test'] = StockDataset(X_test,y_test)
        dataloaders = {x: DataLoader(stock_datasets[x], batch_size=int(config[7]),
                                                     shuffle=False)
                      for x in ['train', 'test']}

        # Train the model
        train_error, test_error = train(model=model, 
                                        dataloaders=dataloaders, 
                                        criterion=loss_function, 
                                        optimizer=optimizer, 
                                        epochs=epochs, 
                                        log_level='Quiet',
                                        save=False,
                                        plot=False,
                                        TRAIN_ON_GPU=TRAIN_ON_GPU,
                                        TRAIN_ON_MULTI_GPUS=TRAIN_ON_MULTI_GPUS)

        # Store the results
        torch.save(model,fpath+str(coun)+'.model')
        final_train_error = float(train_error)
        final_test_error = float(test_error)

        result = [coun,final_train_error, final_test_error] + config.tolist()
        grid_results.append(result)

        # Free memory

        del model
        del loss_function
        del optimizer
        del X_train
        del y_train
        del X_test
        del y_test
        del stock_datasets
        del dataloaders
        del train_error
        del test_error
        gc.collect()
        torch.cuda.empty_cache()

    # Search finished
    grid_time_elapsed = time.time() - grid_since
    print('Grid search complete in {:.0f}m {:.0f}s'.format(
        grid_time_elapsed // 60, grid_time_elapsed % 60))

    #return grid_results

def finegridSearch(grid_results,tickr,n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,batch_size,data, epochs, n_test,n_steps_out,fpath,TRAIN_ON_GPU=False,TRAIN_ON_MULTI_GPUS=False):
    
    # Keep track of time
    grid_since = time.time()

    # Create list of all combinations to search
    configurations = np.array(np.meshgrid(n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,batch_size,epochs)).T.reshape(-1,9)
    print(f'Hyper Parameter search across {len(configurations)} different configurations.')

    # Get a data loader
    dl = MyDataLoader()

    # For each configuration, build a model and train
    coun = 0
    for config in configurations:
        coun +=1
        print(f'Iteration: {coun} |epochs: {int(config[8])}|periods: {int(config[1])}|hidden: {int(config[2])}|layers: {int(config[3])}|drop: {config[4]}|lr: {config[5]}|clip: {int(config[6])}')
        # Build model
        model = StockLSTM(tickr=tickr,
            n_features=int(config[0]),
            n_periods=int(config[1]), 
            n_hidden=int(config[2]), 
            n_layers=int(config[3]), 
            drop_prob=config[4], 
            lr=config[5],
            clip=int(config[6]),
            TRAIN_ON_GPU=TRAIN_ON_GPU,
            TRAIN_ON_MULTI_GPUS=TRAIN_ON_MULTI_GPUS)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config[5])

        # Build dataset
        X_train, y_train, X_test, y_test = dl.train_test_split(data=data.to_numpy(), n_steps_in=int(config[1]), n_steps_out=n_steps_out,n_test=n_test)
        stock_datasets = {}
        stock_datasets['train'] = StockDataset(X_train,y_train)
        stock_datasets['test'] = StockDataset(X_test,y_test)
        dataloaders = {x: DataLoader(stock_datasets[x], batch_size=int(config[7]),
                                                     shuffle=False)
                      for x in ['train', 'test']}

        # Train the model
        train_error, test_error = train(model=model, 
                                        dataloaders=dataloaders, 
                                        criterion=loss_function, 
                                        optimizer=optimizer, 
                                        epochs=int(config[8]), 
                                        log_level='Quiet',
                                        save=False,
                                        plot=False,
                                        TRAIN_ON_GPU=TRAIN_ON_GPU,
                                        TRAIN_ON_MULTI_GPUS=TRAIN_ON_MULTI_GPUS)

        # Store the results
        torch.save(model,fpath+str(coun)+'.model')
        final_train_error = float(train_error)
        final_test_error = float(test_error)

        result = [coun,final_train_error, final_test_error] + config.tolist()
        grid_results.append(result)

        # Free memory

        del model
        del loss_function
        del optimizer
        del X_train
        del y_train
        del X_test
        del y_test
        del stock_datasets
        del dataloaders
        del train_error
        del test_error
        gc.collect()
        torch.cuda.empty_cache()

    # Search finished
    grid_time_elapsed = time.time() - grid_since
    print('Grid search complete in {:.0f}m {:.0f}s'.format(
        grid_time_elapsed // 60, grid_time_elapsed % 60))

def gridSearch2(grid_results,tickr,n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,batch_size,data, epochs, n_test,n_steps_out,fpath,TRAIN_ON_GPU=False,TRAIN_ON_MULTI_GPUS=False):
    
    # Keep track of time
    grid_since = time.time()

    # Create list of all combinations to search
    configurations = np.array(np.meshgrid(n_features,n_periods, n_hidden, n_layers, drop_prob, lr,clip,batch_size)).T.reshape(-1,8)
    print(f'Hyper Parameter search across {len(configurations)} different configurations.')

    # Get a data loader
    dl = MyDataLoader()

    # For each configuration, build a model and train
    coun = 0
    for config in configurations:
        coun +=1
        print(f'Iteration: {coun} |periods: {int(config[1])}|hidden: {int(config[2])}|layers: {int(config[3])}|drop: {config[4]}|lr: {config[5]}|clip: {int(config[6])}')
        # Build model
        model = StockLSTM(tickr=tickr,
            n_features=int(config[0]),
            n_periods=int(config[1]), 
            n_hidden=int(config[2]), 
            n_layers=int(config[3]), 
            drop_prob=config[4], 
            lr=config[5],
            clip=int(config[6]),
            TRAIN_ON_GPU=TRAIN_ON_GPU,
            TRAIN_ON_MULTI_GPUS=TRAIN_ON_MULTI_GPUS)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Build dataset
        X_train, y_train, X_test, y_test = None, None, None, None
        for key, value in data.items():
            if X_train is None:
                X_train, y_train, X_test, y_test = dl.train_test_split(data=value.to_numpy(), n_steps_in=int(config[1]), n_steps_out=n_steps_out,n_test=n_test)
            else:
                X_train2, y_train2, X_test2, y_test2 = dl.train_test_split(data=value.to_numpy(), n_steps_in=int(config[1]), n_steps_out=n_steps_out,n_test=n_test)
                X_train = np.concatenate((X_train,X_train2))
                y_train = np.concatenate((y_train,y_train2))
                X_test = np.concatenate((X_test,X_test2))
                y_test = np.concatenate((y_test,y_test2))

        stock_datasets = {}
        stock_datasets['train'] = StockDataset(X_train,y_train)
        stock_datasets['test'] = StockDataset(X_test,y_test)
        dataloaders = {x: DataLoader(stock_datasets[x], batch_size=int(config[7]),
                                                     shuffle=True)
                      for x in ['train', 'test']}

        # Train the model
        train_error, test_error = train(model=model, 
                                        dataloaders=dataloaders, 
                                        criterion=loss_function, 
                                        optimizer=optimizer, 
                                        epochs=epochs, 
                                        log_level='Quiet',
                                        save=False,
                                        plot=False,
                                        TRAIN_ON_GPU=TRAIN_ON_GPU,
                                        TRAIN_ON_MULTI_GPUS=TRAIN_ON_MULTI_GPUS)

        # Store the results
        torch.save(model,fpath+str(coun)+'.model')
        final_train_error = float(train_error)
        final_test_error = float(test_error)

        result = [coun,final_train_error, final_test_error] + config.tolist()
        grid_results.append(result)

        # Free memory

        del model
        del loss_function
        del optimizer
        del X_train
        del y_train
        del X_test
        del y_test
        del stock_datasets
        del dataloaders
        del train_error
        del test_error
        gc.collect()
        torch.cuda.empty_cache()

    # Search finished
    grid_time_elapsed = time.time() - grid_since
    print('Grid search complete in {:.0f}m {:.0f}s'.format(
        grid_time_elapsed // 60, grid_time_elapsed % 60))

    #return grid_results

def visualizeGridSearch(result_grid,figsize=(10,10)):
    '''model_state = [item[0] for item in result_grid]
    optimizer_state = [item[1] for item in result_grid]
    train_error = [item[2] for item in result_grid]
    test_error = [item[3] for item in result_grid]
    n_features = [item[4] for item in result_grid]
    n_periods = [item[5] for item in result_grid]
    n_hidden = [item[6] for item in result_grid]
    n_layers = [item[7] for item in result_grid]
    drop_prob = [item[8] for item in result_grid]
    lr = [item[9] for item in result_grid]
    clip = [item[10] for item in result_grid]
    batch_size = [item[11] for item in result_grid]
    '''
    model_id = [item[0] for item in result_grid]
    train_error = [item[1] for item in result_grid]
    test_error = [item[2] for item in result_grid]
    n_features = [item[3] for item in result_grid]
    n_periods = [item[4] for item in result_grid]
    n_hidden = [item[5] for item in result_grid]
    n_layers = [item[6] for item in result_grid]
    drop_prob = [item[7] for item in result_grid]
    lr = [item[8] for item in result_grid]
    clip = [item[9] for item in result_grid]
    batch_size = [item[10] for item in result_grid]

    fig, axs = plt.subplots(3,3,figsize=figsize)
    fig.suptitle("Grid Search Results")
    
    axs[0,0].set_title('# Features')
    axs[0,0].scatter(n_features,train_error,label='Train',color='C0')
    axs[0,0].scatter(n_features,test_error,label='Test',color='C1')

    axs[0,1].set_title('# Periods')
    axs[0,1].scatter(n_periods,train_error,label='Train',color='C0')
    axs[0,1].scatter(n_periods,test_error,label='Test',color='C1')
    
    axs[0,2].set_title('# Hidden Nodes')
    axs[0,2].scatter(n_hidden,train_error,label='Train',color='C0')
    axs[0,2].scatter(n_hidden,test_error,label='Test',color='C1')
    
    axs[1,0].set_title('# Hidden Layers')
    axs[1,0].scatter(n_layers,train_error,label='Train',color='C0')
    axs[1,0].scatter(n_layers,test_error,label='Test',color='C1')
    
    axs[1,1].set_title('Drop Probability')
    axs[1,1].scatter(drop_prob,train_error,label='Train',color='C0')
    axs[1,1].scatter(drop_prob,test_error,label='Test',color='C1')
    
    axs[1,2].set_title('Learning Rate')
    axs[1,2].scatter(lr,train_error,label='Train',color='C0')
    axs[1,2].scatter(lr,test_error,label='Test',color='C1')
    
    axs[2,0].set_title('Clip Size')
    axs[2,0].scatter(clip,train_error,label='Train',color='C0')
    axs[2,0].scatter(clip,test_error,label='Test',color='C1')
    
    axs[2,1].set_title('Batch Size')
    axs[2,1].scatter(batch_size,train_error,label='Train',color='C0')
    axs[2,1].scatter(batch_size,test_error,label='Test',color='C1')
    
    axs[2,2].axis('off')
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Train Error',
                          markerfacecolor='C0', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Test Error',
                          markerfacecolor='C1', markersize=10),]
    fig.legend(handles=legend_elements)
    fig.tight_layout()
    plt.show()

def visualizeFineGridSearch(result_grid,figsize=(10,10)):

    batch_size = [item[10] for item in result_grid]
    train_error = [item[1] for item in result_grid]
    test_error = [item[2] for item in result_grid]
    epochs = [item[11] for item in result_grid]

    fig, axs = plt.subplots(1,2,figsize=figsize)
    fig.suptitle("Fine Grid Search Results")
    
    axs[0].set_title('Batch Size')
    axs[0].scatter(batch_size,train_error,label='Train',color='C0')
    axs[0].scatter(batch_size,test_error,label='Test',color='C1')

    axs[1].set_title('Epochs')
    axs[1].scatter(epochs,train_error,label='Train',color='C0')
    axs[1].scatter(epochs,test_error,label='Test',color='C1')
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Train Error',
                          markerfacecolor='C0', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Test Error',
                          markerfacecolor='C1', markersize=10),]
    fig.legend(handles=legend_elements)
    fig.tight_layout()
    plt.show()

def predict(model,input_df,output_df,scaler, n_steps_in, n_steps_out,n_test):
    
    # Set model in evaluation mode
    model.eval()
    
    # Creating a list for storing transformed data
    X = []
    y = []
    df = input_df.to_numpy()
    
    # Create training data
    for i in range(len(df)-n_steps_in):
        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out

        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(df):
            break

        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x,seq_y = df[i:end, :], df[end:out_end, 0]

        X.append(seq_x)
        y.append(seq_y)

    inputs = Variable(torch.Tensor(np.array(X)))
    outputs = Variable(torch.Tensor(np.array(y)))
    
    ds = StockDataset(inputs,outputs)
    dl = DataLoader(ds, batch_size=256,shuffle=False)
    
    preds = []
    
    for inputs,outputs in dl:
        # Get batch size
        batch_size = inputs.size()[0]

        # Evaluate on data
        try:
            h = model.init_hidden(batch_size)
        except AttributeError:
            # if using DataParallel wrapper to use multipl GPUs
            h = model.module.init_hidden(batch_size)

        # if using GPUs, load validation data to device
        if model.TRAIN_ON_GPU or model.TRAIN_ON_MULTI_GPUS:
            inputs = inputs.cuda()
        
        with torch.set_grad_enabled(False):
            outputs, val_h = model(inputs, h)


        preds = preds + outputs.view(-1).tolist()
    
    # Pad predictions
    pred_type = [np.nan for i in range(len(df)-len(preds))] + \
            ['Training' for i in range(len(preds)-n_test)] + \
            ['Test' for i in range(n_test)]
    preds = [np.nan for i in range(len(df)-len(preds))] + preds
    
    
    # Add predictions to df
    df = output_df.copy()
    df['Raw_Predictions'] = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    df['Prediction_Type'] = pred_type
    df['Predicted'] = df['Raw_Predictions']*df['Open']+df['Open']
    df['Predicted'] = df['Predicted'].shift(1)
    #df = df[['Prediction_Type','Open','Predicted']]
    #print(df)
    #df.rename(columns=['Prediction_Type','Actual','Predicted'],inplace=True)
    
    return df

def plotPrediction(df):
    fig, axs = plt.subplots(3,figsize=(16,12))
    fig.suptitle(f"Predicted vs Actual Opening Prices")
    
    axs[0].set_title('Full Series')
    axs[0].set_ylabel('Stock Price ($)')
    axs[0].plot(df[df['Prediction_Type']=='Training']['Predicted'], label='Predicted-Train', color='C0')
    axs[0].plot(df[df['Prediction_Type']=='Test']['Predicted'], label='Predicted-Test', color='C1')
    axs[0].plot(df['Open'], label='Actual', color='C2')
    axs[0].legend(loc='upper left')
    
    axs[1].set_ylabel('Stock Price ($)')
    axs[1].set_xlabel('Date')
    axs[1].set_title('Training Set')
    axs[1].plot(df[df['Prediction_Type']=='Training']['Predicted']['2021-01-01':], label='Predicted', color='C0')
    axs[1].plot(df[df['Prediction_Type']=='Training']['Open']['2021-01-01':], label='Actual', color='C2')
    axs[1].legend(loc='upper left')

    axs[2].set_ylabel('Stock Price ($)')
    axs[2].set_xlabel('Date')
    axs[2].set_title('Test Set')
    axs[2].plot(df[df['Prediction_Type']=='Test']['Predicted'], label='Predicted', color='C1')
    axs[2].plot(df[df['Prediction_Type']=='Test']['Open'], label='Actual', color='C2')
    axs[2].legend(loc='upper left')

    fig.tight_layout()
    plt.show()

def rmse(actual, predicted):
    """
    Calculates the root mean square error between the two arrays
    """
    rms = (actual-predicted)**2

    # Returning the sqaure root of the root mean square
    return float(np.sqrt(rms.mean()))

def mae(actual, predicted):
    """
    Calculates the root mean square error between the two arrays
    """
    rms = np.abs(actual-predicted)

    # Returning the sqaure root of the root mean square
    return float(rms.mean())

def errordirection(actual, predicted):
    error = actual-predicted
    
    poserror = float(error[error >0].mean())
    poserrorp = float(len(error[error >0]) / len(error[error.notna()]))
    negerror = float(error[error <0].mean())
    negerrorp = float(len(error[error <0]) / len(error[error.notna()]))
    
    return (poserror,poserrorp,negerror,negerrorp)
    
    
def evaluateModel(model,input_data, output_data, scaler,n_steps_in, n_steps_out,n_test):
    # Get the predictions
    df = predict(model,input_data, output_data,scaler, n_steps_in, n_steps_out,n_test)
    
    # Calculate statistics
    train_rmse = rmse(df[df['Prediction_Type']=='Training']['Open'],df[df['Prediction_Type']=='Training']['Predicted'])
    test_rmse = rmse(df[df['Prediction_Type']=='Test']['Open'],df[df['Prediction_Type']=='Test']['Predicted'])
    print(f'Training RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')
    
    train_mae = mae(df[df['Prediction_Type']=='Training']['Open'],df[df['Prediction_Type']=='Training']['Predicted'])
    test_mae = mae(df[df['Prediction_Type']=='Test']['Open'],df[df['Prediction_Type']=='Test']['Predicted'])
    print(f'Training MAE: {train_mae}')
    print(f'Test MAE: {test_mae}')
    
    train_errordirect = errordirection(df[df['Prediction_Type']=='Training']['Open'],df[df['Prediction_Type']=='Training']['Predicted'])
    test_errordirect = errordirection(df[df['Prediction_Type']=='Test']['Open'],df[df['Prediction_Type']=='Test']['Predicted'])
    print('Training - Above Error: ${:.2f} ({:.2f}%) Below Error: ${:.2f} ({:.2f}%)'.format(train_errordirect[0],train_errordirect[1]*100,train_errordirect[2],train_errordirect[3]*100))
    print('Test - Above Error: ${:.2f} ({:.2f}%) Below Error: ${:.2f} ({:.2f}%)'.format(test_errordirect[0],test_errordirect[1]*100,test_errordirect[2],test_errordirect[3]*100))
    
    # Plot results
    plotPrediction(df)

    return df


def sortGridResults(result_grid):
  
    result_grid.sort(key = lambda x: x[2])
    return result_grid

def resultDF(result_grid):
    return pd.DataFrame([item[1:] for item in result_grid],columns=[
        'Train Error',
        'Test Error',
        '# Features',
        '# Periods',
        '# Hidden Nodes',
        '# Hidden Layers',
        'Drop Probability',
        'Learning Rate',
        'Clip Size',
        'Batch Size'])  

def resultDFFine(result_grid):
    return pd.DataFrame([item[1:] for item in result_grid],columns=[
        'Train Error',
        'Test Error',
        '# Features',
        '# Periods',
        '# Hidden Nodes',
        '# Hidden Layers',
        'Drop Probability',
        'Learning Rate',
        'Clip Size',
        'Batch Size',
        'Epochs'])  

class Lot:
    def __init__(self, ticker, quantity, price):
        self.ticker = ticker
        self.quantity = quantity
        self.price = price
        self.profit=0
        
class Portfolio:
    def __init__(self, 
                    covariance_matrix,TREASURY_BILL_RATE,
                 starting_balance = 100000, 
                stocks = ['SPY','AAPL','AMC','BB','F','GME','GRWG','MO','NIO','PLTR','RKT','SNDL','TLRY','TSLA','VIAC'],
                expected_returns=None):
        self.cash = starting_balance
        self.stocks = stocks
        self.positions = {stock: [] for stock in stocks}
        self.profit = 0
        self.profits = {stock: 0 for stock in stocks}
        self.hist_bal = []
        self.hist_profit = []
        self.hist_period = []
        self.hist_invested = []
        self.hist_mix = []
        self.spy_prices = []

        self.TREASURY_BILL_RATE = TREASURY_BILL_RATE
        self.asset_expected_returns = None
        with open(covariance_matrix,'rb') as f:
            self.covariance_matrix = pickle.load(f)
        self.weights = self.random_weights(len(self.stocks))
        if expected_returns is not None:
            with open(expected_returns,'rb') as f:
                self.asset_expected_returns = pickle.load(f)
                self.optimize_sharpe_ratio()

        
    
    def setExpectedReturns(self, predictions):
        #self.asset_expected_returns = np.nan_to_num(np.log(predictions).reshape(-1, 1), nan=0)
        self.asset_expected_returns = np.nan_to_num(predictions.reshape(-1, 1), nan=0)
    
    def getBookValue(self, current_prices):
        positions = self.getPositions()
        
        return current_prices @ positions + self.cash
    
    def getPositions(self):
        positions = []
        for stock in self.stocks:
            positions.append(self.getNumShares(stock))
        
        return np.array(positions)
    
    def getPostionValue(self,stock,price):
        numshares = self.getNumShares(stock)
        val = numshares *  price
        return val

    def getNumShares(self,stock):
        lots = [lot.quantity for lot in self.positions[stock]]
        return np.sum(lots)
    
    def buy(self,stock, price, quantity,pprint=True):
        lot = Lot(stock, quantity, price)
        self.positions[stock].append(lot)
        self.cash -= price*quantity
        if pprint:
            print('Bought {} shares of {} at ${:.2f} for a total of ${:.2f}'.format(quantity,stock,price,price*quantity))
        
    def sell(self,stock, price, quantity,pprint=True):
        num_left = quantity
        
        profit = 0
        self.cash += price*quantity
        while (num_left>0):

            # Pop off oldest lot
            current_lot = self.positions[stock].pop(0)
            
            # If lot contains more shares than needed then just
            # reduce number of shares in lot
            if current_lot.quantity > num_left:
                p = (price - current_lot.price) * num_left
                profit += p
                if pprint:
                    print('Sold {} shares of {} at ${:.2f} for a gain/loss of ${:.2f}'.format(num_left,stock,price,p))
                current_lot.quantity -= num_left
                num_left = 0
                self.positions[stock].insert(0,current_lot)
            
            # If lot contains less than needed to sell pop it out
            else:
                p = (price - current_lot.price) * num_left
                profit += p
                if pprint:
                    print('Sold {} shares of {} at ${:.2f} for a gain/loss of ${:.2f}'.format(current_lot.quantity,stock,price,p))
                num_left -= current_lot.quantity
                
        self.profit += profit
        self.profits[stock] += profit
        
    def random_weights(self,weight_count):
        weights = np.random.random((weight_count, 1))
        weights /= np.sum(weights)
        return weights.reshape(-1, 1)

    def unsafe_optimize_with_risk_tolerance(self, risk_tolerance):
        res = minimize(
          lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
          self.random_weights(self.weights.size),
          constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
          ],
          bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_with_risk_tolerance(self, risk_tolerance):
        assert risk_tolerance >= 0.
        return self.unsafe_optimize_with_risk_tolerance(risk_tolerance)

    def optimize_with_expected_return(self, expected_portfolio_return):
        res = minimize(
          lambda w: self._variance(w),
          self.random_weights(self.weights.size),
          constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return},
          ],
          bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_sharpe_ratio(self):
        # Maximize Sharpe ratio = minimize minus Sharpe ratio
        res = minimize(
          lambda w: -(self._expected_return(w) - self.TREASURY_BILL_RATE / 100) / np.sqrt(self._variance(w)),
          self.random_weights(self.weights.size),
          constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
          ],
          bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def _expected_return(self, w):
        return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]

    def _variance(self, w):
        return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

    def rebalancePortfolio(self, period, prices,pprint=True,method='sharpe',expected_portfolio_return=None,risk_tolerance=None):
        # Get current book value
        curr_bal = self.getBookValue(prices)
        print('Current Value: ${:.2f}'.format(curr_bal))

        # Optimize portfolio
        if method == 'expected_portfolio_return':
            self.optimize_with_expected_return(expected_portfolio_return)
        
        elif method == 'risk_tolerance':
            self.optimize_with_risk_tolerance(risk_tolerance)

        else:
            self.optimize_sharpe_ratio()

        # Rebalance
        # get target value of each stock
        target = np.round(self.weights * curr_bal,2).reshape(-1)
        
        # get target number of shares for each stock
        target_shares = target / prices

        for idx, stock in enumerate(self.stocks):
            curr_num_shares = self.getNumShares(stock)
            change = target_shares[idx] - curr_num_shares
            
            if change<0:
                self.sell(stock,prices[idx],change*-1,pprint)
                
            elif change>0:
                self.buy(stock,prices[idx],change,pprint)
                
        
        # Keep track of history
        self.hist_bal.append(curr_bal)
        self.hist_profit.append(self.profit)
        self.hist_period.append(period)
        self.spy_prices.append(prices[0])
        self.hist_invested.append(self.weights.reshape(-1))
    
    def rebalancePortfolioBase(self, period, prices,pprint=True,method='sharpe',expected_portfolio_return=None,risk_tolerance=None):
        # Get current book value
        curr_bal = self.getBookValue(prices)
        print('Current Value: ${:.2f}'.format(curr_bal))

        # Rebalance
        # get target value of each stock
        target = np.round(self.weights * curr_bal,2).reshape(-1)
        
        # get target number of shares for each stock
        target_shares = target / prices

        for idx, stock in enumerate(self.stocks):
            curr_num_shares = self.getNumShares(stock)
            change = target_shares[idx] - curr_num_shares
            
            if change<0:
                self.sell(stock,prices[idx],change*-1,pprint)
                
            elif change>0:
                self.buy(stock,prices[idx],change,pprint)
                
        
        # Keep track of history
        self.hist_bal.append(curr_bal)
        self.hist_profit.append(self.profit)
        self.hist_period.append(period)
        self.spy_prices.append(prices[0])
        self.hist_invested.append(self.weights.reshape(-1))
    
    def rebalancePortfolioOld(self, period, prices, buys,weights=None,pprint=True):
        # Get current book value
        curr_bal = self.getBookValue(prices)
        
        print('Current Value: ${:.2f}'.format(curr_bal))
        
        # Optimize number of shares 
        def objective(w):
            return curr_bal - (prices * buys) @ w[0:-1]
        
        # Create Bounds
        # Creates a tuple of tuples to pass to minimize
        # to ensure all weights are betwen [0, inf]
        non_neg = []
        for i in range(len(prices)):
            non_neg.append((0.,(curr_bal/4)/prices[i]))
        non_neg.append((0,None))
        non_neg = tuple(non_neg)
        
        const = ({'type' : 'eq' , 'fun' : lambda w: curr_bal - w[0:-1] @ (prices*buys) - w[-1] })
        # Run optimization with SLSQP solver
        if buys.sum() == 0:
            w = np.zeros(len(prices)+1)
        elif weights is not None:
            w = weights * buys
            w = w / w.sum() * curr_bal / prices
            w = np.append(w,0)
        else:
            w = buys * (curr_bal / buys.sum()) / prices
            w = np.append(w,0)

        solution = minimize(fun=objective, x0=w, method='SLSQP',constraints=const,bounds=non_neg)

        shares = (solution.x[0:-1]*buys).astype(int)

        for idx, stock in enumerate(self.stocks):
            curr_num_shares = self.getNumShares(stock)
            change = shares[idx] - curr_num_shares
            
            if change<0:
                self.sell(stock,prices[idx],change*-1,pprint)
                
            elif change>0:
                self.buy(stock,prices[idx],change,pprint)
                
        
        # Keep track of history
        self.hist_bal.append(curr_bal)
        self.hist_profit.append(self.profit)
        self.hist_period.append(period)
        self.spy_prices.append(prices[0])
        invested = shares * prices
        invested = np.append(invested, [self.cash])
        invested = invested / curr_bal
        self.hist_invested.append(invested)
    
    def plotHist(self):
        fig, ax1 = plt.subplots(figsize=(16,6))
        bals = pd.Series(self.hist_bal,index=self.hist_period)    
        ax1.set_title('Portfolio Balance Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.plot(bals, color='C0',label='Portfolio Value')
        ax1.tick_params(axis='y')

        plt.show()
        
        
        
    def plotReturns(self):        
        fig, ax = plt.subplots(figsize=(16,9))
        initial_bal = self.hist_bal[0]
        returns = pd.Series(self.hist_bal,index=self.hist_period)
        returns = (returns-initial_bal) / initial_bal * 100
        
        spy = pd.Series(self.spy_prices,index=self.hist_period)
        spy_initial = spy[0]
        spy = (spy - spy_initial) / spy_initial * 100
        ax.set_title('Portfolio Performance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')  
        ax.plot(returns, color='C1',label='Portfolio Returns')
        ax.plot(spy, color='C2',label='S&P500')
        
        fig.legend()
        plt.show()


    def plotSharpeReturns(self):        
        fig, ax = plt.subplots(figsize=(16,9))
        initial_bal = self.hist_bal[0]
        returns = pd.Series(self.hist_bal,index=self.hist_period)
        returns = (returns-initial_bal) / initial_bal * 100
        sharpe_retuns = [returns[:i].mean() / returns[:i].std() for i in range(len(returns))]
        
        spy = pd.Series(self.spy_prices,index=self.hist_period)
        spy_initial = spy[0]
        spy = (spy - spy_initial) / spy_initial * 100
        ax.set_title('Portfolio Performance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')  
        ax.plot(returns, color='C1',label='Portfolio Returns')
        ax.plot(spy, color='C2',label='S&P500')
        
        fig.legend()
        plt.show()
        
        
    def plotInvestment(self):        
        fig, ax = plt.subplots(figsize=(16,9))
        invested = pd.DataFrame(self.hist_invested,index=self.hist_period,columns=self.stocks)
        invested = invested * 100
        
        invested.plot.bar(stacked=True,ax=ax)
        ax.set_title('Portfolio Mix')
        plt.show()
        
    def comparePortfolios(self,other):
        initial_bal = self.hist_bal[0]
        returns = pd.Series(self.hist_bal,index=self.hist_period)
        returns = (returns-initial_bal) / initial_bal * 100
        
        initial_bal2 = other.hist_bal[0]
        returns2 = pd.Series(other.hist_bal,index=other.hist_period)
        returns2 = (returns2-initial_bal2) / initial_bal2 * 100
        
        fig, ax1 = plt.subplots(figsize=(16,6))
        #fig.suptitle(a + ' vs ' + b)

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Returns %')
        ax1.plot(returns, color='C0',label='Current Strategy')
        ax1.plot(returns2, color='C1',label='Base Strategy')

        fig.legend()
        plt.show()
        
class PaperTrader:
    # Constructor
    # id: api id for accont
    # key: secret api key for account
    def __init__(self, ID, key, covariance_matrix,TREASURY_BILL_RATE,live=False):
        # Connection Info
        self.apiID = ID
        self.apiKey = key
        if live:
            self.base_url = 'https://api.alpaca.markets'
        else:
            self.base_url = 'https://paper-api.alpaca.markets'
        self.data_url = 'https://data.alpaca.markets'
        
        # Setup Alpaca API Connection
        self.api = tradeapi.REST(
            self.apiID,
            self.apiKey,
            self.base_url
        )
        
        # Get account info
        self.account = self.api.get_account()
        
        # Get positions
        self.positions = self.api.list_positions()
        
        # Stocks available to trade
        self.stocks = ['SPY','AAPL','AMC','BB','F','GME','GRWG','MO','NIO','PLTR','RKT','SNDL','TLRY','TSLA','VIAC']

        self.TREASURY_BILL_RATE = TREASURY_BILL_RATE
        self.asset_expected_returns = None
        with open(covariance_matrix,'rb') as f:
            self.covariance_matrix = pickle.load(f)
        self.weights = self.random_weights(len(self.stocks))
    
    def setExpectedReturns(self, predictions):
        #self.asset_expected_returns = np.nan_to_num(np.log(predictions).reshape(-1, 1), nan=0)
        self.asset_expected_returns = np.nan_to_num(predictions.reshape(-1, 1), nan=0)
    
    def getPerformance(self):
        hist = self.api.get_portfolio_history()
        base_value = hist._raw['base_value']
        phist = pd.Series(hist._raw['equity'],index=hist._raw['timestamp'])
        phist.index = pd.to_datetime(phist.index, unit='s')
        up = (phist[-1] - base_value) >= 0
        return phist, up

    def getAccountInfo(self):
        self.account = self.api.get_account()
        return self.account
    
    def getBookValue(self):
        a = self.getAccountInfo()
        return float(a.portfolio_value)
    
    def getPositions(self):
        self.positions = self.api.list_positions()
        return self.positions
    
    def getNumShares(self, tickr):
        try:
            qty = int(self.api.get_position(tickr).qty)
        except:
            qty = int(0)
        return qty
    
    def getPositionValue(self, tickr):
        try:
            value = float(self.api.get_position(tickr).market_value)
        except:
            value = 0
        return value

    def buy(self, tickr,quantity=None,notional=None):
        if quantity is not None:
            print('Buying {} shares of {}.'.format(quantity,tickr))
            self.api.submit_order(symbol=tickr,
                             qty=quantity, 
                              side="buy", 
                              type="market", 
                              time_in_force="day", 
                              limit_price=None)
        if notional is not None:
            
            try:
                self.api.submit_order(symbol=tickr,
                             notional=notional, 
                              side="buy", 
                              type="market", 
                              time_in_force="day", 
                              limit_price=None)
                print('Buying ${} worth of {}.'.format(notional,tickr))
            except:
                price = self.getStockPrice(tickr)
                quantity = int(notional / price)
                if quantity >= 1:
                    print('Buying {} shares of {}.'.format(quantity,tickr))
                    self.api.submit_order(symbol=tickr,
                                 qty=quantity, 
                                  side="buy", 
                                  type="market", 
                                  time_in_force="day", 
                                  limit_price=None)

    def sell(self, tickr,quantity=None,notional=None):
        if quantity is not None:
            print('Selling {} shares of {}.'.format(quantity,tickr))
            self.api.submit_order(symbol=tickr,
                             qty=quantity, 
                              side="sell", 
                              type="market", 
                              time_in_force="day", 
                              limit_price=None)
        if notional is not None:
            try:
                self.api.submit_order(symbol=tickr,
                             notional=notional, 
                              side="sell", 
                              type="market", 
                              time_in_force="day", 
                              limit_price=None)

                print('Selling ${} worth of {}.'.format(notional,tickr))
            except:
                price = self.getStockPrice(tickr)
                quantity = int(notional / price)
                if quantity >= 1:
                    print('Selling {} shares of {}.'.format(quantity,tickr))
                    self.api.submit_order(symbol=tickr,
                                 qty=quantity, 
                                  side="sell", 
                                  type="market", 
                                  time_in_force="day", 
                                  limit_price=None)
    
    def getStockOpenPrice(self, tickr):
        return float(self.api.get_bars(tickr,
                                 tradeapi.rest.TimeFrame.Minute, 
                                 datetime.datetime.now().date(),
                                 datetime.datetime.now().date(),
                                 limit=1,adjustment="raw")._raw[0]['o'])
    def getStockPrice(self, tickr):
        return float(self.api.get_bars(tickr,
                                 tradeapi.rest.TimeFrame.Minute, 
                                 datetime.datetime.now().date(),
                                 datetime.datetime.now().date(),
                                 limit=1,adjustment="raw")._raw[0]['c'])
    def getOpeningPrices(self):
        prices = []
        for stock in self.stocks:
            res = self.getStockOpenPrice(stock)
            prices.append(res)
        return prices
    
    def random_weights(self,weight_count):
        weights = np.random.random((weight_count, 1))
        weights /= np.sum(weights)
        return weights.reshape(-1, 1)

    #https://medium.com/analytics-vidhya/modern-portfolio-theory-model-implementation-in-python-e416facabf46
    def unsafe_optimize_with_risk_tolerance(self, risk_tolerance):
        res = minimize(
          lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
          self.random_weights(self.weights.size),
          constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
          ],
          bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_with_risk_tolerance(self, risk_tolerance):
        assert risk_tolerance >= 0.
        return self.unsafe_optimize_with_risk_tolerance(risk_tolerance)

    def optimize_with_expected_return(self, expected_portfolio_return):
        res = minimize(
          lambda w: self._variance(w),
          self.random_weights(self.weights.size),
          constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return},
          ],
          bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_sharpe_ratio(self):
        # Maximize Sharpe ratio = minimize minus Sharpe ratio
        res = minimize(
          lambda w: -(self._expected_return(w) - self.TREASURY_BILL_RATE / 100) / np.sqrt(self._variance(w)),
          self.random_weights(self.weights.size),
          constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
          ],
          bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def _expected_return(self, w):
        return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]

    def _variance(self, w):
        return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

    def rebalancePortfolio(self, method='sharpe',expected_portfolio_return=None,risk_tolerance=None):
        # Get current book value
        curr_bal = self.getBookValue()
        print('Current Value: ${:.2f}'.format(curr_bal))

        # Optimize portfolio
        if method == 'expected_portfolio_return':
            self.optimize_with_expected_return(expected_portfolio_return)
        
        elif method == 'risk_tolerance':
            self.optimize_with_risk_tolerance(risk_tolerance)

        else:
            self.optimize_sharpe_ratio()

        # Rebalance
        # get target value of each stock
        target = np.round(self.weights * curr_bal,2).reshape(-1)

        # First sell necessary shares
        for idx, stock in enumerate(self.stocks):
            curr_value = self.getPositionValue(stock)
            change = int(target[idx] - curr_value)
            
            if change<0:
                self.sell(stock,notional=change*-1)

        # Then buy shares
        for idx, stock in enumerate(self.stocks):
            curr_value = self.getPositionValue(stock)
            change = int(target[idx] - curr_value)
            
            if change>0:
                self.buy(stock,notional=change)



    def rebalancePortfolioOld(self, prices,buys,weights=None):
        # Get current book value
        curr_bal = self.getBookValue()
        print('Current Value: ${:.2f}'.format(curr_bal))
        
        # Optimize number of shares 
        def objective(w):
            return curr_bal - (prices * buys) @ w[0:-1]
        
        # Create Bounds
        non_neg = []
        for i in range(len(prices)):
            non_neg.append((0.,(curr_bal/4)/prices[i]))
        non_neg.append((0,None))
        non_neg = tuple(non_neg)
        
        const = ({'type' : 'eq' , 'fun' : lambda w: curr_bal - w[0:-1] @ (prices*buys) - w[-1] })
        
        # Run optimization with SLSQP solver
        if buys.sum() == 0:
            w = np.zeros(len(prices)+1)
        elif weights is not None:
            w = weights * buys
            w = w / w.sum() * curr_bal / prices
            w = np.append(w,0)
        else:
            w = buys * (curr_bal / buys.sum()) / prices
            w = np.append(w,0)

        solution = minimize(fun=objective, x0=w, method='SLSQP',constraints=const,bounds=non_neg)

        shares = (solution.x[0:-1]*buys).astype(int)
        print(shares)
        # First sell necessary shares
        for idx, stock in enumerate(self.stocks):
            curr_num_shares = self.getNumShares(stock)
            change = int(shares[idx] - curr_num_shares)
            
            if change<0:
                self.sell(stock,change*-1)

        # Then buy shares
        for idx, stock in enumerate(self.stocks):
            curr_num_shares = self.getNumShares(stock)
            change = int(shares[idx] - curr_num_shares)
            
            if change>0:
                self.buy(stock,change)


# Plot todays predictions
def plotTodaysPrediction(df,filename=None,margin=0.4,text_margin=0.02):
    pos_color = '#00ff41'
    pos_border = '#00ff41'
    neg_color = '#F21A1D'
    neg_border = '#F21A1D'
    text_color = 'white'
    background_color = 'black'
    fontsize = 15
    text_spacing = text_margin
    n_lines = 10
    diff_linewidth = 1
    alpha_value = 0.03

    max_val = df['Change'].abs().max() + margin

    fig, ax = plt.subplots(figsize=(15,10))
    for n in range(1, n_lines+1):
        ax.barh(df[df['Change']>0].index,df[df['Change']>0]['Change'],align='center',color=pos_color,edgecolor=pos_border,linewidth=2+(diff_linewidth*n),alpha=alpha_value)
    ax.barh(df[df['Change']>0].index,df[df['Change']>0]['Change'],align='center',color=pos_color)
    for n in range(1, n_lines+1):
        ax.barh(df[df['Change']<=0].index,df[df['Change']<=0]['Change'],align='center',color=neg_color,edgecolor=neg_border,linewidth=2+(diff_linewidth*n),alpha=alpha_value)
    ax.barh(df[df['Change']<=0].index,df[df['Change']<=0]['Change'],align='center',color=neg_color)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.invert_yaxis()
    #ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_facecolor(background_color)
    ax.set_xlim([-max_val,max_val])
    for idx, row in df.iterrows():
        if row['Change'] > 0:
            ax.text(0-text_spacing, idx, row['Stock'], ha='right', fontsize=fontsize,color = text_color)
            ax.text(row['Change']+text_spacing, idx, "{:.2f}%".format(row['Change']), ha='left', fontsize=fontsize,color = text_color)
        else:
            ax.text(0+text_spacing, idx, row['Stock'], ha='left', fontsize=fontsize,color = text_color)
            ax.text(row['Change']-text_spacing, idx, "{:.2f}%".format(row['Change']), ha='right', fontsize=fontsize,color = text_color)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def plotPerformance(df, start,up,filename=None):
    filter_df = df[start:]

    plt.style.use("seaborn-dark")
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = 'black'#'#212946'  # bluish dark grey
    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'  # very light grey
    background_color = 'black'

    colors = [
        '#08F7FE',  # teal/cyan
        '#FE53BB',  # pink
        '#F5D300',  # yellow
        '#00ff41',  # matrix green
    ]
    if up:
        color = colors[3]
    else:
        color = colors[1]

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(10,5))

    filter_df.plot(marker='o', color=color, ax=ax)

    n_shades = 10
    diff_linewidth = 1.05
    alpha_value = 0.3 / n_shades

    for n in range(1, n_shades+1):
        filter_df.plot(marker='o',
                linewidth=2+(diff_linewidth*n),
                alpha=alpha_value,
                legend=False,
                ax=ax,
                color=color)
    orig_xlim = ax.get_xlim()
    orig_ylim = ax.get_ylim()
    # Color the areas below the lines:
    ax.fill_between(x=filter_df.index,
                        y1=filter_df,
                        color=color,
                        alpha=0.1)
    ax.grid(color='#2A3459')
    ax.set_xlim(orig_xlim)
    ax.set_ylim(orig_ylim)
    ax.yaxis.set_major_formatter('${x:,.0f}')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()