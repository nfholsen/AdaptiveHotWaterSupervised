import numpy as np
import pandas as pd

def lagged_data_preparation(EHW):
    
    EHW = EHW
    
    one_week_lag = 24*7
    
    lagged_data = []
    
    for t in range(len(EHW)):
        
        end_row = one_week_lag + t
        
        if end_row > len(EHW) -1:
            
            break
            
        _1H = end_row-1
        _2H = end_row-2
        _12H = end_row-12
        _1D = end_row-24
        _1W = t
        
        row =   [EHW[_1W]] + [EHW[_1D]] + [EHW[_12H]] + [EHW[_2H]] + [EHW[_1H]] + [EHW[end_row]]
                                                                                               
        lagged_data.append(row)
                                                    
    return np.array(lagged_data)

def rolling_hourly_mean_EHW(df_,household_id): 
    N = 30
    alpha = 1/N

    dates = [date.strftime("%Y-%m-%d") for date in df_.index[0::24]]
        
    df_['rolling_mean'] = 0
             
    for i, date in enumerate(dates):
                                  
        if i<=N and i>0:
            df_.loc[date,'rolling_mean'] = (df_.loc[dates[i-1],'rolling_mean'].values*(i-1) + df_.loc[dates[i-1],str(household_id)].values)/i
        else:
            df_.loc[date,'rolling_mean'] = df_.loc[dates[i-1],'rolling_mean'].values*(1-alpha) + df_.loc[dates[i-1],str(household_id)].values*alpha

    return np.reshape(df_['rolling_mean'].values,(-1,1))

def daily_hourly_rolling_mean_EHW(df_,household_id):
    
    N = 4
    alpha = 1/N
    
    daily_hourly_rl = []
    
    for i in range(len(df_)):
        
        if i < 24*7:
            
            daily_hourly_rl.append(df_[str(household_id)][i-24*7])
            
        elif i >= 24*7 and i<N*24*7:
            daily_hourly_rl.append((int(i/7*24)*daily_hourly_rl[i-24*7]+df_[str(household_id)][i-24*7])/(int(i/7*24)+1))
        else:  
            daily_hourly_rl.append(daily_hourly_rl[i-24*7]*(1-alpha)+df_[str(household_id)][i-24*7]*alpha)
            
    return np.array(daily_hourly_rl).reshape(-1,1)

def is_consumption_last12h(df_,household_id):
    
    datetimes = [date.strftime("%Y-%m-%d %H:%M:%S") for date in df_.index]
    
    for i in range(len(datetimes)):
        
        if i < 12:    
            if i == 0:       
                df_.loc[datetimes[i],'consumption_last12H'] = 0      
            else:  
                df_.loc[datetimes[i],'consumption_last12H'] = int(sum(df_.loc[datetimes[0]:datetimes[i],str(household_id)])>0)             
        else:     
             df_.loc[datetimes[i],'consumption_last12H'] = int(sum(df_.loc[datetimes[i-12]:datetimes[i],str(household_id)])>0)
                
    return np.reshape(df_['consumption_last12H'].values,(len(df_),1))

def is_consumption_last24h(df_,household_id):
    
    datetimes = [date.strftime("%Y-%m-%d %H:%M:%S") for date in df_.index]
    
    for i in range(len(datetimes)):
        
        if i < 24:    
            if i == 0:       
                df_.loc[datetimes[i],'consumption_last24H'] = 0      
            else:  
                df_.loc[datetimes[i],'consumption_last24H'] = int(sum(df_.loc[datetimes[0]:datetimes[i],str(household_id)])>0)             
        else:     
             df_.loc[datetimes[i],'consumption_last24H'] = int(sum(df_.loc[datetimes[i-24]:datetimes[i],str(household_id)])>0)
                
    return np.reshape(df_['consumption_last24H'].values,(len(df_),1))

def part_of_consumption(df_,household_id):
    
    """
    daily_EHW: percentage of daily consumption for each hour with consumption
    """
    
    dates = [date.strftime("%Y-%m-%d") for date in df_.index[0::24]]
    proportion = []
    
    for date in dates:
        
        daily_EHW = df_.loc[date,str(household_id)].values
        if np.sum(daily_EHW) > 0:
            proportion.extend(daily_EHW*100/np.sum(daily_EHW))
        else: proportion.extend([0]*24)
    return np.array(proportion)

def prepare_output(df_,household_id,method='percentage'):
    
    """
    Prepare the multi-class output.
    y_null = 0 => no consumption or low consumption
    y_med = 1 => normal consumption
    y_high = 2 => high consumption
    
    """
    
    EHW = df_[str(household_id)].values
    
    if method =='percentage':
        
        proportion = part_of_consumption(df_,household_id)  
        threshold_1 = 2
        threshold_2 = 20
        y =  (proportion>threshold_1).astype(int) + (proportion>threshold_2).astype(int)
        
    elif method =='quantile':
        
        quantiles = np.quantile(EHW[EHW>0],[0.25,0.75])
        y = (EHW>quantiles[0]).astype(int) + (EHW>quantiles[1]).astype(int)
        
    else:
    
        print('Choose either quantile or percentage for the method argument')
        
    return np.array(y)

def binary_consumption(df,house_number):
    """
    Warning : will return the modified dataframe, not an array to append
    """
    cnsumption = (df[f'{house_number}']>0).astype(int)
    return cnsumption

def prepare_dataset(df_,household_id):
    
    EHW = lagged_data_preparation(df_[str(household_id)])
    
    features = ['EHW_1W','EHW_1D','EHW_12H','EHW_2H','EHW_1H']
    
    X = EHW[:,:-1]
        
    PROPORTION = lagged_data_preparation(part_of_consumption(df_,household_id))[:,:2]
    X = np.concatenate((X,PROPORTION),axis=1)
        
    features.extend(['PROP_1W','PROP_1D'])
        
    WORKDAY = df_['workday'][7*24:].values
    WORKDAY = np.reshape(WORKDAY,(len(WORKDAY),1))
    X=np.concatenate((X,WORKDAY),axis=1)
        
    features.append('WORKDAY')

    HOLIDAYS = df_['holidays'][7*24:].values
    HOLIDAYS = np.reshape(HOLIDAYS,(len(HOLIDAYS),1))
    X=np.concatenate((X,HOLIDAYS),axis=1)
        
    features.append('HOLIDAYS')
      
    WEEKDAY = df_.loc[:,'day_0':'day_5'][7*24:].values
    X = np.concatenate((X,WEEKDAY),axis=1)
        
    features.extend(['DAY0','DAY1','DAY2','DAY3','DAY4','DAY5'])
        
        
    X = np.concatenate((X,rolling_hourly_mean_EHW(df_,household_id)[7*24:]),axis=1)
    X = np.concatenate((X,daily_hourly_rolling_mean_EHW(df_,household_id)[7*24:]),axis=1)
        
    features.extend(['HOURLY_MEAN','DAILY_HOURLY_MEAN'])
        
        
    X = np.concatenate((X,is_consumption_last24h(df_,household_id)[7*24:]),axis=1)
    X = np.concatenate((X,is_consumption_last12h(df_,household_id)[7*24:]),axis=1)
        
    features.extend(['IS_CONSUMPTION_LAST24H','IS_CONSUMPTION_LAST12H'])
        
        
    Y = EHW[:,-1]
    
    return X, Y, features
