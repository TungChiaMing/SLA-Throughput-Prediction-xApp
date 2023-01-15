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


"""Ken QP xApp ML Enhanced , import library
"""
#from statsmodels.tsa.api import VAR
#from statsmodels.tsa.stattools import adfuller
import joblib 
from statsmodels.tsa.api import VAR, ARMA
from statsmodels.tsa.stattools import adfuller, pacf
import pandas as pd
import numpy as np
import scipy.stats as st

class DataNotMatchError(Exception):
    pass

"""OSC E-Relase QP ML PROCESS

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
    for name, column in df.iteritems():# adfuller_test() is using the souce code from osc ric-app-qp
      res_adf.append(self.adfuller_test(column))  # Perform ADF test
    self.diff = 0
    if not all(res_adf):

      self.stationary_data = df.diff().dropna() # take the first difference 
      self.reverse.append(self.stationary_data) # store the reverse data, used for transforming to the orignal scale
 
      res_adf_processed = self.adfuller_test(self.stationary_data)
 
      self.diff =  self.diff + 1 
      # take the first difference until the data is stationary
      while not res_adf_processed:
        self.stationary_data = self.stationary_data.diff().dropna()
        self.reverse.append(self.stationary_data)
        self.diff = self.diff + 1
 
        res_adf_processed = self.adfuller_test(self.stationary_data)
      
 
      self.diff_time[cell] = self.diff # store the times we take difference for a cell
    else:
      self.stationary_data = df
      self.diff_time[cell] = self.diff
       
  def invert_transformation(self, inp, forecast, cell, check_error=False):
    """
    Revert back the differencing to get the forecast to original scale.
    since we have take first difference to get stationary iteratively
    when we're reverting back, still the same
    """

    self.diff = self.diff_time[cell]
  
    if self.diff == 0:
      df_n = forecast.copy()
      df_n.index = pd.date_range(start=inp.index[-1], freq='10ms', periods=len(df_n))  
      return df_n

    df_n = forecast.copy()
    while self.diff >= 2:
      reverse_inp = self.reverse[self.diff - 2] # e.g. double diff = self.reverse[1], when diff =3, we need self.reverse[1]
      if check_error is True:
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

      df_n[col] = pd.concat([drop_inverse_row,df_n[col]],axis=0).astype(int)
      if check_error is True:
        df_n.index = pd.date_range(start=inp.index[0], freq='10ms', periods=len(df_n))
      else:
        df_n.index = pd.date_range(start=inp.index[-1], freq='10ms', periods=len(df_n))   

    self.diff = self.diff - 1  

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

    duplicates_checker = self.data.drop_duplicates()
    if len(duplicates_checker) == 1:
      #print("====================Not Predictable !=======================")
      return False
    else:
    
      self.make_stationary(cell)

      self.pacf_calculate()
      return True
  def pacf_calculate(self):
    """ Calculate the PACF to get the model lag candidate
    """
    self.cell_lag = len(self.stationary_data) // 2 -1

    self.pacf_list = pacf(self.stationary_data,nlags=self.cell_lag) # call pacf() to get every pacf of lag

    avg, dev = self.stationary_data.mean(), self.stationary_data.std()
    df_norm = (self.stationary_data - avg) / dev

    # calculate 0.95 confidence interval to know which lag have the strong correlation with the current situation
    confidence_interval_array = st.norm.interval(alpha=0.95,
                  loc=np.mean(df_norm),
                  scale=st.sem(df_norm))
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

    # if the order of lag have  the strong correlation with the current situation , store it as the model candidate
    for i in self.pacf_list:
      ar_order = ar_order + 1
      if i!=1 and abs(i)>=self.confidence_interval:
        self.ar_orders.append(ar_order)
      

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

      # use BIC to calculate the which model lag shall be selected
      try:
        for ar_order in self.ar_orders:
        #print('BIC for AR(%s): %s'%(ar_order, fitted_model_dict[ar_order].bic)) bic of 
          model_fit_lag_orders[ar_order] = fitted_model_dict[ar_order].bic
        temp = min(model_fit_lag_orders.values())
        res = [key for key in model_fit_lag_orders if model_fit_lag_orders[key] == temp]
      except:
        """
        TODO:  Can't build a few orders : SVD did not converge  
        """  
        res.append(self.cell_lag) 
    else:
      """
      TODO:  what if we can't find an order in the 0.95 confidence interval ?
        --> does it means the training data isn't enough ?
      
      """
      res.append(self.cell_lag)
    return res[0]




"""Ken QP xApp ML Enhanced
training flow has depicted in the website below 
https://hackmd.io/NvYiLkJ9SvONbUWnis2RbQ?view#PDCP-Data-Volumn-Prediction
"""
def train(db, cid):
    """
        fit An AR model for a cell by using the optimal lag
    """
    training_data_len = 0
    db.read_data(meas='QP', cellid=cid)

    md = PROCESS(db.data) # 1. adf test & stationary
    predictable = md.process(cid)
    
    training_data_len = len(md.data)
    if predictable:
        md.pacf_calculate()  # 2. calculate pacf
        md.ar_model_candidate() # 3. select model candidate
        optimal_lag = md.find_optimal_lag() # 3. model selection
        try:  
            model = ARMA(md.stationary_data, order=(optimal_lag,0))  # fit & save model 
            model_fit = model.fit() 
        except: # if SVD did not converge
            # Dummy Build
            #model = ARMA(md.stationary_data, order=(2,0))
            #model_fit = model.fit()
            for i in range(optimal_lag):
                optimal_lag = optimal_lag - 1
                try:  
                    model = ARMA(md.stationary_data, order=(optimal_lag,0)) # do 4.    
                    model_fit = model.fit()    
                    if model_fit is not None:
                        break
                except: # if SVD did not converge
                    pass
    
        file_name = 'qp/'+cid.replace('/', '')
        try:
            with open(file_name, 'wb') as f:
                joblib.dump(model_fit, f)     # Save the model with the cell id name
        except:
            print("!!!!!!!!!!!!!!!!!!Warning: You don't have the model!!!!!!!!!!!!")
    return training_data_len
