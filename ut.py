import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import math
import numpy as np

import datetime as dt

import plotly.offline as pyo
import plotly

import plotly.graph_objects as go
from plotly.subplots import make_subplots

pyo.init_notebook_mode(connected=True)
pd.options.plotting.backend = 'plotly'
# Ch1 Functions
      
# Ch 2
    
def buy_n_hold(p_df = df_price, stocks = stocks, mode = 'value', total_investment = 8000, brokerage_fee = 20):
    price_df = p_df
    df = pd.DataFrame(index = p_df.index)
    new_df = df
    
    for stock in stocks:
        initial = price_df[stock][0]
        final = price_df[stock][-1]
        step = (final - initial) / len(new_df)
        value_array = []
        percentage_array = []

        if total_investment != None :
            stock_num = (total_investment - brokerage_fee) / initial
            for day_index in range(len(price_df[stock])):
                current_value = (initial + step * day_index) 
                value_array.append(current_value * stock_num)
                percentage_array.append(((current_value / initial) - 1) * 100)
        elif total_investment == None:
            for day_index in range(len(price_df[stock])):
                current_value = initial + step * day_index
                value_array.append(current_value)
                percentage_array.append(((current_value / initial) - 1) * 100)

        if mode == 'value':
            new_df[stock] = value_array
        elif mode == 'percentage':
            new_df[stock] = percentage_array
    return new_df
bnh = buy_n_hold(p_df = df_price, mode = 'percentage', total_investment = 8000, stocks = stocks)
#final_results(bnh, output = True)
#bnh.plot()

def doll_cost_av(p_df = df_price, day_freq = 60, trade_amount = 5000, stocks = stocks, brokerage_fee = 20, 
                 mode = 'value'):
    new_df = pd.DataFrame(index = p_df.index)
    price_df = p_df
    inv_amount = 0
    inv_array = []
    inv_counter = 0
    for time_index in price_df[stocks].index:
        # trade on the first day
        if len(inv_array) == 0:
            inv_amount += trade_amount
            inv_array.append(inv_amount)
            inv_counter += 1
        elif inv_counter != day_freq: 
            inv_array.append(inv_amount)
            inv_counter += 1
        elif inv_counter == day_freq:
            inv_amount += trade_amount
            inv_array.append(inv_amount)
            inv_counter = 0
    #print(inv_array)
    new_df['invested'] = inv_array

    # dollar cost average
    for stock in stocks:
        price_amount = 0
        price_array = []
        day_counter = 0
        stock_amount = 0
        value_delta_array = []
        for time_index in price_df[stocks].index:
            if len(price_array) == 0:
                price_amount += trade_amount
                #print(stock,time_index,price_df[stock][time_index])
                stock_amount += (trade_amount- brokerage_fee) / price_df[stock][time_index]
                price_array.append(price_amount)
                value_delta_array.append(0)
                day_counter += 1
            elif day_counter != day_freq: 
                price_array.append(stock_amount * price_df[stock][time_index])
                value_delta_array.append(stock_amount * price_df[stock][time_index] - new_df['invested'][time_index])
                day_counter += 1
            elif day_counter == day_freq:
                price_amount += trade_amount
                stock_amount += (trade_amount- brokerage_fee) / price_df[stock][time_index]
                price_array.append(stock_amount * price_df[stock][time_index])
                value_delta_array.append(stock_amount * price_df[stock][time_index] - new_df['invested'][time_index])
                day_counter = 0
            #print(price_amount)
            
        if mode == 'percentage':
            new_df[stock] = ((price_array / new_df['invested']) -1 )* 100 
        elif mode == 'value':
            new_df[stock] = price_array
        elif mode == 'value_delta':
            new_df[stock] = value_delta_array
    if mode == 'percentage' or mode == 'value_delta':
        new_df.drop(columns=['invested'], inplace=True)
    
    return new_df

dca = doll_cost_av(p_df = df, mode = 'percentage', stocks = stocks)
#final_results(dca, output = True)
#dca.plot()

#Ch 3.1 

# Normalises using a rolling median
def norm_df_med(df = df_price, window = 30, stocks = stocks):
    new_df = pd.DataFrame()
    for stock in stocks:
        new_df[stock] = (df[stock] / (df[stock].rolling(window, min_periods = 0).median()) - 1)
    return new_df

# Normalises using rolling mean
def norm_df_mean(df = df_price, window = 30,  stocks = stocks):
    new_df = pd.DataFrame()
    for stock in stocks:
        new_df[stock] = df[stock] / (df[stock].rolling(window, min_periods = 0).mean()-1)
    return new_df
gradient = norm_df_mean(df = df_price, window = 30,  stocks = stocks)
# Inputs: normalised stock prices df
# Outputs: area under the graph with a reduction factor df

def norm_sum_df(df = gradient , reduction_factor = 1 , stocks = stocks):
    new_df = pd.DataFrame(index = df.index)
    for stock in stocks:
        totals = []
        rolling_sum = 0
        for time_index in df.index:
            rolling_sum = (rolling_sum * reduction_factor + df[stock][time_index])
            totals.append(rolling_sum)
        new_df[stock] = totals
    return new_df
integral = norm_sum_df(df = gradient, reduction_factor = 0.9, stocks = stocks)
#integral.plot()

# Input: 2 dfs, the first being the transformed df and another being the df with pricing
#        can also input upper and lower threshholds

# Output: df that determines to trade or not based on threshholds
#         also outputs stock price for each given day

def decision_1(integral = integral , df_price = df_price , stocks = stocks, 
               up_thresh = 0.9, low_thresh = 0.25, thresh_window = 180, trig_thresh = 20):
    decisions = pd.DataFrame(index = integral.index)
    for stock in stocks:
        #trim_df = df[df.index > end - start]
        trim_df = integral
        n_df = integral
        #trim_df_price = df_price[df_price.index > end - start]
        trim_df_price = df_price
        #n_df = (trim_df)

        new_df = pd.DataFrame(index = integral.index)
        new_df['price'] = df_price[stock]
        new_df['norm'] = integral[stock]
        new_df['up_bound'] = integral[stock].rolling(thresh_window, min_periods = 0).quantile(up_thresh).copy()
        new_df['low_bound'] = integral[stock].rolling(thresh_window, min_periods = 0).quantile(low_thresh).copy()

        trade = []
        buy_trig, sell_trig = 0, 0   
        cooler_counter = 0
        for time_index in integral[stock].index:
            if integral[stock][time_index] > new_df['up_bound'][time_index] and sell_trig == 0 and cooler_counter > 10:
                sell_trig = 1
                trade.append(1)
            elif integral[stock][time_index] < new_df['low_bound'][time_index] and buy_trig == 0 and cooler_counter > 10:
                buy_trig = 1
                trade.append(-1)
            else:
                trade.append(0)
            cooler_counter+=1

            #Triggers for sell
            if sell_trig < trig_thresh and sell_trig > 0:
                sell_trig += 1
            elif sell_trig == trig_thresh:
                sell_trig = 0

            #Triggers for buy
            if buy_trig < trig_thresh and buy_trig > 0:
                buy_trig += 1
            elif buy_trig == trig_thresh:
                buy_trig = 0 

        new_df['trade'] = trade
        decisions[stock] = trade
    return decisions
decide_1 = decision_1(integral = integral, df_price = df_price, stocks = stocks,up_thresh = 0.6, low_thresh = 0.4, thresh_window = 180,
                   trig_thresh = 1)
#decide.plot()

def trade_price_df(price_df = df_price , decision_trade_df = decide_1 , brokerage_fee = 20, 
                   trade_amount = 6000, compound_factor = None , bank = 8000, mode = 'percentage',
                  stocks = stocks):
    bank_initial = bank
    print('Trade Output:', mode)
    #print(decision_trade_df)
    new_df = pd.DataFrame(index = decision_trade_df.index)

    #if trade_amount > 10000:
    #    brokerage_fee = 30
    for stock in stocks:
        bank = bank_initial
        if compound_factor != None :
            trade_amount = compound_factor * bank
        holding_stocks = 0
        bank_value = []
        total_value = []
        for date_index in decision_trade_df[stock].index:
            if (decision_trade_df[stock][date_index] == -1) & (bank >= trade_amount): 
                # print('buy')
                bank += - trade_amount
                holding_stocks += (trade_amount - brokerage_fee) / price_df[stock][date_index]
                if compound_factor != None :
                    trade_amount = compound_factor * bank
            elif (decision_trade_df[stock][date_index] == 1) and (holding_stocks > 0):
                # print('sell')
                bank += holding_stocks * price_df[stock][date_index] - brokerage_fee
                holding_stocks = 0
                if compound_factor != None :
                    trade_amount = compound_factor * bank
            bank_value.append(bank)
            if math.isnan(price_df[stock][date_index]) == True:
                total_value.append(bank +  holding_stocks * price_df[stock][date_index])
            else:
                total_value.append(bank + holding_stocks * price_df[stock][date_index])
        last_index = -1
        while math.isnan(price_df[stock][last_index]) == True:
            last_index += -1
        else:
            bank_value[-1] = bank + holding_stocks * price_df[stock][last_index]
            #print(stock,bank_value[-21],bank_value[-1])
        
        if mode == 'value':
            new_df[stock] = total_value
        elif mode == 'bank':
            new_df[stock] = bank_value
        elif mode == 'percentage':
            new_df[stock] = ((np.array(total_value) / bank_initial) -1) * 100
    return(new_df)
price_1 = trade_price_df(price_df = df_price, decision_trade_df = decide_1, stocks = stocks, mode = 'percentage',
                         brokerage_fee = 20, trade_amount = 6000, 
                         compound_factor = 1, bank = 8000)
#price_1.plot()

### Ch 3.2

def norm_int_df(norm_df = gradient , stocks = stocks):
    new_df = pd.DataFrame(index = norm_df.index)
    for stock in stocks:
        crossing_array = []
        for time_index in range(len(norm_df[stock])):
            if time_index == 0:
                crossing_array.append(0)
            elif norm_df[stock][time_index - 1] < 0 and norm_df[stock][time_index] >= 0:
                # indicate to buy
                crossing_array.append(-1)
            elif norm_df[stock][time_index - 1] >= 0 and norm_df[stock][time_index] < 0:
                # indicate to sell
                crossing_array.append(1)
            else: 
                crossing_array.append(0)
        new_df[stock] = crossing_array
    return new_df
intercept = norm_int_df(gradient, stocks = stocks)
#intercept.plot()   


# Input: intercept indicator df and normalised area df
# Output: trade df 
# Gradient = 0, check area graph if its large enough, then buy/sell

def decision_2(norm_int_df =intercept , norm_sum_df = integral , limit_factor = 1, stocks = stocks,
                     trig_thresh = 20):
    new_df = pd.DataFrame(index = norm_int_df.index)
    for stock in stocks:
        trade_array = []
        pos_limit, neg_limit = [],[]
        upper_limit, lower_limit = 0,0
        buy_trig, sell_trig, cooler_counter = 0, 0, 0
        for time_index in range(len(norm_int_df)):
            if norm_sum_df[stock][time_index] > 0:
                pos_limit.append(norm_sum_df[stock][time_index])
                upper_limit = np.mean(pos_limit)
            elif norm_sum_df[stock][time_index] < 0:
                neg_limit.append(norm_sum_df[stock][time_index])
                lower_limit = np.mean(neg_limit)
            #print(norm_int_df[stock][time_index],norm_sum_df[stock][time_index] - (limit_factor * upper_limit))
            #if (norm_sum_df[stock][time_index] > (limit_factor * upper_limit)):#and sell_trig == 0and cooler_counter > 5:
                #print(norm_int_df[stock][time_index],norm_sum_df[stock][time_index] - (limit_factor * upper_limit))
            if norm_int_df[stock][time_index] == 1 and (norm_sum_df[stock][time_index] > (limit_factor * upper_limit)) and sell_trig == 0 and cooler_counter > 5:
                #upper_limit = norm_sum_df[stock][time_index]
                #print(stock,'FIRE')
                trade_array.append(1)
                sell_trig = 1
            elif norm_int_df[stock][time_index] == -1 and (norm_sum_df[stock][time_index] < (limit_factor * lower_limit)) and buy_trig == 0 and cooler_counter > 5:
                #lower_limit = norm_sum_df[stock][time_index]
                #print('tick')
                trade_array.append(-1)
                buy_trig = 1
            else:
                trade_array.append(0)
            
            cooler_counter += 1
               
            #Triggers for sell
            if sell_trig < trig_thresh and sell_trig > 0:
                sell_trig += 1
            elif sell_trig == trig_thresh:
                sell_trig = 0

            #Triggers for buy
            if buy_trig < trig_thresh and buy_trig > 0:
                buy_trig += 1
            elif buy_trig == trig_thresh:
                buy_trig = 0 
        new_df[stock] = trade_array
    return(new_df)
decide_2 = decision_2(norm_int_df = intercept, norm_sum_df = integral, stocks = stocks, limit_factor = 0.32)
#decide_2.plot()

### Ch 3.3
def integral_bound(integral = integral , stocks = stocks, 
                   high_ratio=0.99, low_ratio= 0.99, sell_with = 0.2, buy_with = 0.2, sell_against = 0.02, buy_against = 0.02):
    new_df = pd.DataFrame(index = integral.index)
    for stock in stocks:
        sell_array = []
        sell_trig = 0
        buy_array = []
        buy_trig = 0
        #high_ratio, low_ratio =0.8, 0.99
        window = 3
        #area_array = [] 
        #area = 0
        for time_index in range(len(integral[stock])):
            # Integral: Positive 
            if integral[stock][time_index] >= 0:
                # Graph > sell_trig
                if integral[stock][time_index] > sell_trig:
                    # Graph going up
                    if time_index != 0 and integral[stock][time_index] > integral[stock][time_index - 1]:
                        sell_trig = (sell_trig * high_ratio  + (integral[stock][time_index])* (sell_with))
                    # Graph going down
                    else:
                        sell_trig = sell_trig * high_ratio + ((integral[stock][time_index])* (sell_with))
                # Graph < sell_trig
                else:#  time_index != 0 and integral[stock][time_index -1] < integral[stock][time_index]:
                    sell_trig = sell_trig * high_ratio + ((integral[stock][time_index])* (sell_against))
                    #print(time_index)
                buy_trig = buy_trig * low_ratio - ((buy_against)*integral[stock][time_index])
            # Integral: Negative
            else:
                # Graph < buy_trig
                if integral[stock][time_index] < buy_trig:
                    # Integral going up
                    if time_index != 0 and integral[stock][time_index] < integral[stock][time_index - 1]:
                        buy_trig = (buy_trig * high_ratio  + integral[stock][time_index]*(buy_with))
                    else:
                        buy_trig = buy_trig * high_ratio + ((integral[stock][time_index])* (buy_against))
                else:#if  time_index != 0 and integral[stock][time_index - 1] > integral[stock][time_index]:
                    buy_trig = buy_trig * high_ratio
                sell_trig = sell_trig * low_ratio - ((sell_against) * integral[stock][time_index])
            sell_array.append(sell_trig)
            buy_array.append(buy_trig)
            
            #area = rolling_ratio * area +  integral[stock][time_index]
            #area_array.append(area)
        new_df[stock] = integral[stock]
        new_df[stock+'_sell_trig'] = sell_array
        new_df[stock+'_buy_trig'] = buy_array
        new_df[stock+'_sell_trig'] = new_df[stock+'_sell_trig'].rolling(window, min_periods = 0).mean()
        new_df[stock+'_buy_trig'] = new_df[stock+'_buy_trig'].rolling(window, min_periods = 0).mean()
    return new_df

integral_b = integral_bound(integral = integral, stocks = stocks)
#print(d)
#integral_b.plot()

def decision_3(integral_bound = integral_b , stocks = stocks, trig_thresh = 20):
    new_df = pd.DataFrame(index = integral_bound.index)
    for stock in stocks:
        trade_array = []
        buy_trig, sell_trig = 0, 0
        for time_index in integral_bound[stock].index:
            if integral_bound[stock][time_index] < integral_bound[stock+'_buy_trig'][time_index] and buy_trig == 0:
                trade_array.append(-1)
                buy_trig = 1
            elif  integral_bound[stock][time_index] > integral_bound[stock+'_sell_trig'][time_index] and sell_trig == 0:
                trade_array.append(1)
                sell_trig = 1
            else:
                trade_array.append(0)
            
            #Triggers for sell
            if sell_trig < trig_thresh and sell_trig > 0:
                sell_trig += 1
            elif sell_trig == trig_thresh:
                sell_trig = 0

            #Triggers for buy
            if buy_trig < trig_thresh and buy_trig > 0:
                buy_trig += 1
            elif buy_trig == trig_thresh:
                buy_trig =0
        new_df[stock] = trade_array
    return new_df
decide_3 = decision_3(integral_bound = integral_b, stocks = stocks, trig_thresh = 1)
#decide_3.plot()

### Ch 3.4

def moving_mean(df_price = df_price , stocks = stocks, mode = 'mean', window = 50):
    new_df = pd.DataFrame(index = df_price.index)
    if mode == 'mean':
        for stock in stocks:
            new_df[stock] = df_price[stock].rolling(window, min_periods = 0).mean()
    elif mode == 'median':
        for stock in stocks:
            new_df[stock] = df_price[stock].rolling(window, min_periods = 0).median()
    return new_df
moving = moving_mean(df_price = df, stocks = stocks, mode = 'mean', window = 50)
#moving.plot()

def moving_mean(df_price = df_price , stocks = stocks, mode = 'mean', window = 50):
    new_df = pd.DataFrame(index = df_price.index)
    if mode == 'mean':
        for stock in stocks:
            new_df[stock] = df_price[stock].rolling(window, min_periods = 0).mean()
    elif mode == 'median':
        for stock in stocks:
            new_df[stock] = df_price[stock].rolling(window, min_periods = 0).median()
    return new_df
moving = moving_mean(df_price = df, stocks = stocks, mode = 'mean', window = 50)
#moving.plot()

def moving_bounds(df_price = df_price, moving_df = moving, stocks = stocks, deviations = 2, mode = 'mean', window = 50):
    new_df = pd.DataFrame(index = df_price.index)
    for stock in stocks:
        new_df[stock+'_price'] = df_price[stock]
        new_df[stock+'_moving_low'] = moving_df[stock]
        #new_df[stock+'_moving_high'] = moving_mean(df_price = df, stocks = [stocks], mode = 'mean',window = 20)[stock]
        new_df[stock+'_up_b'] = moving_df[stock] + deviations * df_price[stock].rolling(window, min_periods = 0).std()
        new_df[stock+'_low_b'] = moving_df[stock] - deviations * df_price[stock].rolling(window, min_periods = 0).std()
    return new_df
moving_b = moving_bounds(moving_df = moving, window = 50, deviations = 1.75, mode = 'mean')
#moving_b.plot()

def decision_4(df_price = df_price , moving_b = moving_b , stocks = stocks, trig_thresh = 20):
    new_df = pd.DataFrame(index = df_price.index)
    for stock in stocks:
        decide_array = []
        buy_trig, sell_trig = 0, 0
        for time_index in df_price[stock].index:
            if df_price[stock][time_index] > moving_b[stock+'_up_b'][time_index] and sell_trig == 0:
                decide_array.append(1)
                sell_trig = 1
            elif df_price[stock][time_index] < moving_b[stock+'_low_b'][time_index] and buy_trig == 0:
                decide_array.append(-1)
                buy_trig = 1
            else:
                decide_array.append(0)
                
            #Triggers for sell
            if sell_trig < trig_thresh and sell_trig > 0:
                sell_trig += 1
            elif sell_trig == trig_thresh:
                sell_trig = 0

            #Triggers for buy
            if buy_trig < trig_thresh and buy_trig > 0:
                buy_trig += 1
            elif buy_trig == trig_thresh:
                buy_trig =0
        new_df[stock] = decide_array
    return new_df
decide_4 = decision_4(df_price = df, moving_b = moving_b, stocks = stocks, trig_thresh = 20)
#print(decide_4)

##### Ch 4: Volatility
# Return a list of stocks sorted by their Sharpe Ratio
def sharpe_list(df = df_price , ascend = True, stock_num = None , days = 252, window_days = 90):
    df_bound = df[:days]
    log_returns = np.log(df_bound/df_bound.shift(1)).dropna()
    daily_std = log_returns.std()
    annualised_vol = daily_std * np.sqrt(252) * 100
    volatility = log_returns.rolling(window = window_days).std()*np.sqrt(window_days)
    Rf = 0.01/252
    sharpe_ratio = (log_returns.rolling(window = window_days).mean() - Rf*window_days/volatility)
    return list(sharpe_ratio.median().sort_values(ascending = ascend)[:stock_num].index)
sharpe_list(df = df_price, ascend = True, stock_num = 4)
                      
def sortino_list(df = df_price  , ascend = True, stock_num = None , days = 252, window_days = 90):
    df_bound = df[:days]
    log_returns = np.log(df_bound/df_bound.shift(1)).dropna()
    daily_std = log_returns.std()
    annualised_vol = daily_std * np.sqrt(252) * 100
    volatility = log_returns.rolling(window = window_days).std()*np.sqrt(window_days)
    Rf = 0.01/252
    sortino_vol = (log_returns[log_returns<0].rolling(window = window_days, center = True, min_periods = 10).mean() - Rf*window_days/volatility)
    sortino_ratio = (log_returns.rolling(window = window_days).mean() - Rf*window_days/sortino_vol)
    return list(sortino_ratio.median().sort_values(ascending = ascend)[:stock_num].index)
sortino_list(df = df_price, ascend = True, stock_num = 4)

##### Ch 5: Ensemble Algorithms
def trade_price_all_df(price_df  = df_price, decision_trade_df = decide_1 , brokerage_fee = 20, 
                   trade_amount = 6000, compound_factor = None  , bank = 8000, mode = 'percentage',
                  stocks = stocks, decide_thresh = 1):
    bank_initial = bank
    print('Trade Output:', mode)
    new_df = pd.DataFrame(index = decision_trade_df.index)

    #if trade_amount > 10000:
    #    brokerage_fee = 30
    for stock in stocks:
        bank = bank_initial
        if compound_factor != None:
            trade_amount = compound_factor * bank
        holding_stocks = 0
        bank_value = []
        total_value = []
        for date_index in decision_trade_df[stock].index:
            if (decision_trade_df[stock][date_index] <= -decide_thresh) & (bank >= trade_amount): 
                # print('buy')
                bank += - trade_amount
                holding_stocks += (trade_amount - brokerage_fee) / price_df[stock][date_index]
                if compound_factor != None:
                    trade_amount = compound_factor * bank
            elif (decision_trade_df[stock][date_index] >= decide_thresh) and (holding_stocks > 0):
                # print('sell')
                bank += holding_stocks * price_df[stock][date_index] - brokerage_fee
                holding_stocks = 0
                if compound_factor != None:
                    trade_amount = compound_factor * bank
            bank_value.append(bank)
            
            if math.isnan(price_df[stock][date_index]) == True:
                total_value.append(bank)
            else:
                total_value.append(bank + holding_stocks * price_df[stock][date_index])
        last_index = -1
        while math.isnan(price_df[stock][last_index]) == True:
            last_index += -1
        else:
            bank_value[-1] = bank + holding_stocks * price_df[stock][last_index]
            #print(stock,bank_value[-21],bank_value[-1])
        
        if mode == 'value':
            new_df[stock] = bank_value
        elif mode == 'bank':
            new_df[stock] = total_value
        elif mode == 'percentage':
            new_df[stock] = ((np.array(total_value) / bank_initial) -1) * 100
    return(new_df)

# Displays

# Output an array [Coefficient of Variation, Mean, Std]
def final_results(df = bnh, stocks = stocks, output = False):
    final_array = []
    for stock in stocks:
        final_array.append(round(df[stock][-1],5))
    if output == True:
        print('Stocks: ',stocks)
        print('Results:',final_array)
        print('Average: {} \261 {}, C_Var: {}'.format(round(np.mean(final_array),2), round(np.std(final_array),2),round(np.mean(final_array)/np.std(final_array),2)))
    return [round(np.mean(final_array)/np.std(final_array),2), round(np.mean(final_array),2), round(np.std(final_array),2)]

def compare_results(df_1 = bnh, df_2 = price_1, stocks = stocks, mode = 'percentage'):
    df_1_results = []
    df_2_results = []
    for stock in stocks:
        df_1_results.append(round(df_1[stock][-1],6))
        df_2_results.append(round(df_2[stock][-1],6))
    print('Stocks name:', stocks)
    print('Algorithm 1:', df_1_results)
    print('Algorithm 2:', df_2_results)
    stats_comparison = []
    for i in range(len(df_1_results)):
        stats_comparison.append(round(df_1_results[i] - df_2_results[i],6))
    print('Difference :', stats_comparison)
    if mode == 'percentage':
        print('Average    : {} \261 {}%, C_Var: {}'.format(round(np.mean(stats_comparison),2), round(np.std(stats_comparison),2), round(np.mean(stats_comparison)/np.std(stats_comparison),2)))  
    elif mode == 'value':
        print('Average    : ${}, C_Var: {}'.format(round(np.mean(stats_comparison),2), round(np.mean(stats_comparison)/np.std(stats_comparison),2)))
#compare_results(df_1 = price_2, df_2 = bnh, stocks = stocks, mode = 'percentage') 

def risk_stats(decide = decide_1 , df_price = df_price , stocks = stocks, mode = 'value', 
              brokerage_fee = 20, trade_amount = 6000, compound_factor = None, bank = 8000, output = False):
    wallet_df = pd.DataFrame(index = decide.index)
    value_price = trade_price_df(price_df = df_price, decision_trade_df = decide, stocks = stocks, mode = mode,
                   brokerage_fee = brokerage_fee, trade_amount = trade_amount, 
                   compound_factor = compound_factor, bank = bank)
    total_array = []
    for time_index in wallet_df.index:
        total_array.append(sum(sum(value_price[value_price.index == time_index].values)))
    wallet_df['running_total'] = total_array
    wallet_df['running_total'].plot()
    fin = wallet_df['running_total'][-1]/(bank*len(stocks))
    most = wallet_df['running_total'].max()/(bank*len(stocks))
    least = wallet_df['running_total'].min()/(bank*len(stocks))
    if output == True:
        print('Statistics for over-all')
        print('Finishing percentage: {}%'.format(round((fin-1)*100,2)))
        print('Highest Percentage:   {}%'.format(round((most-1)*100,2)))
        #print('Lowest Percentage:   {}%'.format(round((least-1)*100,2)))
    return(wallet_df)

#wallet = risk_stats(decide = decide_2, df_price = df_price, output = True,
#                    compound_factor = compound_factor, bank = bank, trade_amount = amount_per_trade, 
#                    brokerage_fee = brokerage_fee)
1
#wallet.plot()

def plotting_results(df_price = df_price, integral = integral, decide = decide_1 , price = price_1 , 
                     stocks = stocks, mode = 'percentage', wallet_mode = 'bank'):
    over_all = risk_stats(decide = decide, df_price = df_price, compound_factor = 1, output = False, stocks = stocks, mode = wallet_mode)
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS
    k=0
    fig_results = make_subplots(rows = 5, cols = 1, shared_xaxes = True,
                       vertical_spacing = 0.05, subplot_titles = ('Bank Stocks','Normalised Stocks', 'Decisions','Performance','Wallet'),
                       row_width = [1,1,1,1,1])
    fig_results.update_layout(title_text="Graphing",
                  title_font_size=21, autosize = False, width = 800, height = 1000)
    for stock in stocks:
        fig_results.add_trace(go.Scatter(x = df_price.index, y = df_price[stock], name = stock, 
                                           legendgroup = stock, marker=dict(color=cols[k])), row =1, col = 1)
        fig_results.add_trace(go.Scatter(x = df_price.index, y = integral[stock], name = stock, 
                                           showlegend = False, legendgroup = stock, marker=dict(color=cols[k])), row =2, col = 1)
        fig_results.add_trace(go.Scatter(x = df_price.index, y = decide[stock], name = stock, 
                                           showlegend = False, legendgroup = stock, marker=dict(color=cols[k])), row =3, col = 1)
        fig_results.add_trace(go.Scatter(x = df_price.index, y = df_price[stock], name = stock, 
                                           showlegend = False, legendgroup = stock, marker=dict(color=cols[k])), row =4, col = 1)
        k += 1
    fig_results.add_trace(go.Scatter(x = df_price.index, y = over_all['running_total'], name = 'wallet', 
                                   legendgroup = 'wallet', marker=dict(color=cols[k])), row =5, col = 1)
    final_results(df = price, stocks = stocks, output = True)
    fig_results.show()
#plotting_results(df_price = df_price, integral = integral, decide = decide_1, price = price_1, stocks = stocks)
#plotting_results(df_price = df_price, integral = integral, decide = decide_2, price = price_2, stocks = stocks)

def compare_graphs(df1 = bnh , df2 = price_1 , stocks = stocks):
    fig = make_subplots(rows = len(stocks), cols = 1, shared_xaxes = False,
                   subplot_titles = (stocks))
    fig.update_layout(title_text="Comparisons",
                  title_font_size=21, autosize = False, width = 800, height = len(stocks) * 120)
    stock_index = 1
    fig.update_annotations(font_size=15)
    for stock in stocks:
        fig.add_trace(go.Scatter(x=df1.index, y = df1[stock], marker=dict(color = "RoyalBlue"), name = 'df1_'+stock, 
                      legendgroup = stock, showlegend = False), row = stock_index, col = 1)
        fig.add_trace(go.Scatter(x=df1.index, y = df2[stock], marker=dict(color = 'red'), name = 'df2_'+ stock, 
                      legendgroup = stock, showlegend = False), row = stock_index, col = 1)
        stock_index += 1
    compare_results(df_1 = df1, df_2 = df2, stocks = stocks, mode = 'percentage') 
    return(fig.show())
#compare_graphs(df1 = price_1, df2 = bnh, stocks = stocks)