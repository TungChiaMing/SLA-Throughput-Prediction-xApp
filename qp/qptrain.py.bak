# ==================================================================================
#  Copyright (c) 2020 HCL Technologies Limited.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==================================================================================


"""Ken QP xApp ML Enhanced"""
#from statsmodels.tsa.api import VAR
#from statsmodels.tsa.stattools import adfuller
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf
import pandas as pd
import numpy as np
import scipy.stats as st

class DataNotMatchError(Exception):
    pass

"""
class PROCESS(object):

    def __init__(self, data):
        self.diff = 0
        self.data = data

    def adfuller_test(self, series, thresh=0.05, verbose=False):
    
        r = adfuller(series, autolag='AIC')
        output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
        p_value = output['pvalue']
        if p_value <= thresh:
            return True
        else:
            return False

    def make_stationary(self):
   
        df = self.data.copy()
        res_adf = []
        for name, column in df.iteritems():
            res_adf.append(self.adfuller_test(column))  # Perform ADF test
        if not all(res_adf):
            self.data = df.diff().dropna()
            self.diff += 1

    def invert_transformation(self, inp, forecast):
 
        if self.diff == 0:
            return forecast
        df = forecast.copy()
        columns = inp.columns
        for col in columns:
            df[col] = inp[col].iloc[-1] + df[col].cumsum()
        self.diff = 0
        return df

    def process(self):
  
        df = self.data.copy()
        try:
            df = df[['pdcpBytesDl', 'pdcpBytesUl']]
        except DataNotMatchError:
            print('Parameters pdcpBytesDl, pdcpBytesUl does not exist in provided data')
            self.data = None
        self.data = df.loc[:, (df != 0).any(axis=0)]
        self.make_stationary()  # check for Stationarity and make the Time Series Stationary

    def valid(self):
        val = False
        if self.data is not None:
            df = self.data.copy()
            df = df.loc[:, (df != 0).any(axis=0)]
            if len(df) != 0 and df.shape[1] == 2:
                val = True
        return val
"""

"""Ken QP xApp ML Enhanced"""
class PROCESS(object):
  def __init__(self, data):
    self.diff = 0
    self.diff_time = {}
    self.reverse = []
    self.data = data
    self.cell_lag = 0
    self.pacf_list = []
    self.stationary_data = None
    self.confidence_interval = 0
    self.ar_orders = []
  def adfuller_test(self, series, thresh=0.05, verbose=False):
      """ADFuller test for Stationarity of given series and return True or False"""
      r = adfuller(series, autolag='AIC')
      output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
      p_value = output['pvalue']
      #print("====================adf p value=======================")
      if p_value <= thresh:
          return True
      else:
          return False

  def make_stationary(self, cell):
    """ Take first difference until data is stationary 
    """
    df = self.data.copy()

    # Do ADF Test in every columnm , adfuller_test will retrun a bool
    res_adf = []
    for name, column in df.iteritems():

      # adfuller_test() is using the souce code from osc ric-app-qp
      res_adf.append(self.adfuller_test(column))  # Perform ADF test
    self.diff = 0
    if not all(res_adf):
      #print("====================adf isn't pass=======================")
      #print(res_adf)
      #avg, dev = df.mean(), df.std()
      #df_norm = (df - avg) / dev
      self.stationary_data = df.diff().dropna() # take the first difference 
      self.reverse.append(self.stationary_data) # store the reverse data, used for transforming to the orignal scale
      #plt.figure(figsize=(10,4))
      #plt.plot(self.stationary_data)
      #plt.title('====================Processed NR Cell throughput=======================', fontsize=18)
      
      #df_norm_diff_Volat = df_norm_diff.groupby(df_norm_diff.index).std(ddof=0)
      #df_norm_diff_Season = df_norm_diff.groupby(df_norm_diff.index).mean().dropna()
      
      res_adf_processed = self.adfuller_test(self.stationary_data)
      #print("====================adf after processed 1 =======================")
      #print(res_adf_processed)
      self.diff =  self.diff + 1

      # take the first difference until the data is stationary 
      while not res_adf_processed:
        self.stationary_data = self.stationary_data.diff().dropna()
        self.reverse.append(self.stationary_data)
        self.diff = self.diff + 1
        #print("====================adf after processed %d====================" %self.diff)
        #plt.figure(figsize=(10,4))
        #plt.plot(self.stationary_data)
        #plt.title('====================Processed NR Cell throughput====================' , fontsize=18)
        res_adf_processed = self.adfuller_test(self.stationary_data)
      
        #print(res_adf_processed) 
      #print("============================Debug===============================================")
      #print(self.diff) 
      self.diff_time[cell] = self.diff # store the times we take difference for a cell
    else:
      self.stationary_data = df
      self.diff_time[cell] = self.diff
       
  def invert_transformation(self, inp, forecast, cell, check_error=False):
    """
    Revert back the differencing to get the forecast to original scale.
    check_error is used for developers to check the error of the prediction
    """
    #print("============================Debug===============================================")
    self.diff = self.diff_time[cell]
    #print(self.diff)  
    if self.diff == 0:
      df_n = forecast.copy()
      df_n.index = pd.date_range(start=inp.index[-1], freq='10ms', periods=len(df_n))  
      return df_n

    df_n = forecast.copy()
    while self.diff >= 2:
      reverse_inp = self.reverse[self.diff - 2] # e.g. double diff = self.reverse[1], when diff =3, we need self.reverse[1]
      if check_error is True:
        #df_n_index = pd.date_range(start=reverse_inp.index[-1], freq='10ms', periods=2) 
        #df_n.index = pd.date_range(start=reverse_inp.index[1], freq='10ms', periods=len(df_n))
        pass
      else:
        df_n.index = pd.date_range(start=reverse_inp.index[-1], freq='10ms', periods=len(df_n))        
      columns = reverse_inp.columns

      df_n = df_n[columns].astype(int)
      for col in columns:

        if check_error is True:
          df_n[col] = df_n[col].cumsum() + reverse_inp[col].iloc[0]
        else:
          df_n[col] = df_n[col].cumsum() + reverse_inp[col].iloc[-1]
        drop_inverse_row = reverse_inp[col].head(1)
        df_n[col] = pd.concat([drop_inverse_row,df_n[col]],axis=0).astype(int)
      

      #plt.figure(figsize=(10,4))
      #plt.plot(df_n)
      #plt.title('=================================Inverse Debug=================================', fontsize=18)
      #plt.xlabel('millisecond', fontsize=10)
      self.diff = self.diff - 1 
    if check_error is True:
      df_n.index = pd.date_range(start=inp.index[1], freq='10ms', periods=len(df_n))
    else:
      df_n.index = pd.date_range(start=inp.index[-1], freq='10ms', periods=len(df_n))   
    columns = inp.columns

    df_n = df_n[columns].astype(int)
    for col in columns:
      if check_error is True:
        df_n[col] = df_n[col].cumsum() + inp[col].iloc[0]
      else:
        df_n[col] = df_n[col].cumsum() + inp[col].iloc[-1]
      drop_inverse_row = inp[col].head(1)
      #print("=====================Debug :  drop_inverse============================================")
      #print(drop_inverse_row)
      df_n[col] = pd.concat([drop_inverse_row,df_n[col]],axis=0).astype(int)
      if check_error is True:
        df_n.index = pd.date_range(start=inp.index[0], freq='10ms', periods=len(df_n))
      else:
        df_n.index = pd.date_range(start=inp.index[-1], freq='10ms', periods=len(df_n))   
    #plt.plot(df_n)
    #plt.title('=================================Original Data=================================', fontsize=18)
    #plt.xlabel('millisecond', fontsize=10)    
    
    self.diff = self.diff - 1  

    #print("===================Check Diff equals to zero!====================")
    #print(self.diff)
    
    return df_n

  def process(self, cell):
    """ Filter throughput parameters, call make_stationary() to check for Stationarity time series
    """
    df = self.data.copy()
    try:
        df = df[['pdcpBytesDl']]
    except DataNotMatchError:
        print('Parameters pdcpBytesDl, pdcpBytesUl does not exist in provided data')
        self.data = None
    self.data = df


    #print("====================Original Data=======================")
    #print(self.data)

    #plt.figure(figsize=(10,4))
    #plt.plot(self.data)
    #plt.title('====================Original NR Cell throughput====================' , fontsize=18)
    duplicates_checker = self.data.drop_duplicates()
    if len(duplicates_checker) == 1:
      #print("====================Not Predictable !=======================")
      return False
    else:
    
      self.make_stationary(cell)
      #print("====================Processed Data=======================")
      #print(self.stationary_data)
      self.pacf_calculate()
      return True
  def pacf_calculate(self):
    """ Calculate the PACF
    """
    self.cell_lag = len(self.stationary_data) // 2 -1
    #print("debug cell lag", self.cell_lag)
    #plot_pacf(self.stationary_data, lags=self.cell_lag)
    #plt.show()
    #print("====================Pacf Value=======================")
    self.pacf_list = pacf(self.stationary_data,nlags=self.cell_lag) # call pacf() to get every pacf of lag
    #print(self.pacf_list)
    avg, dev = self.stationary_data.mean(), self.stationary_data.std()
    df_norm = (self.stationary_data - avg) / dev
    confidence_interval_array = st.norm.interval(alpha=0.95,
                  loc=np.mean(df_norm),
                  scale=st.sem(df_norm)) # calculate 0.95 confidence interval to know which lag have the strong correlation with the current situation
    self.confidence_interval = abs(confidence_interval_array[0][0])  
  def valid(self):
    val = False
    if self.data is not None:
        df = self.data.copy()
        df = df.loc[:, (df != 0).any(axis=0)]
        if len(df) != 0 and df.shape[1] == 2:
            val = True
    return val
  def ar_model_candidate(self):
    """ To pick out lag that have strong correlation with the current situation
    """
    ar_order = -1
    for i in self.pacf_list:
      ar_order = ar_order + 1
      if i!=1 and abs(i)>=self.confidence_interval: 
        self.ar_orders.append(ar_order) # if the order of lag have  the strong correlation with the current situation , store it as the model candidate
      

  def find_optimal_lag(self):

    """ Determine the max lag considered
    """
    res = []
    if len(self.ar_orders) != 0 : 
      fitted_model_dict = {}
      try:
        for idx, ar_order in enumerate(self.ar_orders):
      #create AR(p) model
        
          ar_model = ARIMA(self.stationary_data, order=(ar_order,0))
          ar_model_fit = ar_model.fit()
          fitted_model_dict[ar_order] = ar_model_fit
      except:
        """
        TODO:  Can't build a few orders : SVD did not converge  
        """  
        pass        
      model_fit_lag_orders = {}
      try:
        for ar_order in self.ar_orders:
        #print('BIC for AR(%s): %s'%(ar_order, fitted_model_dict[ar_order].bic)) bic of 
          model_fit_lag_orders[ar_order] = fitted_model_dict[ar_order].bic
        temp = min(model_fit_lag_orders.values())
        res = [key for key in model_fit_lag_orders if model_fit_lag_orders[key] == temp]
      except:
        """
        TODO:  Can't build a few orders :　SVD did not converge  
        """  
        res.append(self.cell_lag) 
      #print("===================================Find Optimal Lag==========================================")
      #print(model_fit_lag_orders)
      
      

    else:
      """
      TODO:  what if we can't find an order in the 0.95 confidence interval ?
        --> does it means the training data isn't enough ?
      
      """
      res.append(self.cell_lag)
    return res[0]




"""Ken QP xApp ML Enhanced"""
def train(db, cid):
    """
        fit An AR model for a cell by using the optimal lag
    """
    training_data_len = 0
    db.read_data(meas='QP', cellid=cid)

    md = PROCESS(db.data) # do 1.
    predictable = md.process(cid)
    
    training_data_len = len(md.data)
    if predictable:
        md.pacf_calculate() # do 2.
        md.ar_model_candidate() # do 2.
        optimal_lag = md.find_optimal_lag() # do 3.
        try:
            model = ARIMA(md.stationary_data, order=(optimal_lag,0))    # do 4.  
            model_fit = model.fit() 
        # if SVD did not converge  , use another best lag 
        except:
            for i in range(optimal_lag):
                optimal_lag = optimal_lag - 1
                try:
                    model = ARIMA(md.stationary_data, order=(optimal_lag,0))    # do 4.
                    model_fit = model.fit()    
                    if model_fit is not None:
                        break
                except:
                    pass
                
         
    
    file_name = 'qp/'+cid.replace('/', '')
    try:
        with open(file_name, 'wb') as f:
            joblib.dump(model_fit, f)     # Save the model with the cell id name
    except:
        pass
    return training_data_len
