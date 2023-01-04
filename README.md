Modification below is based on osc ric-app-qp branch e-release  https://github.com/o-ran-sc/ric-app-qp

## Rule must be follow
- 1. The dummy data file needs to be a csv file, supposed to be generated from RIC Test
- 2. Types in Cell Metrics shall have :
    - `measTimeStampRf`
    - `nrCellIdentity`
    - `pdcpBytesDl`
## About date worten into InfluxDB

- 1. `pdcpBytesDl` is the original data
- 2. `pdcpBytesDl_predict` is the predicted data
- 3.  data point is `QP` , and measurement is `QP_Prediction` (you can change the name in `insert.py` as you prefer)
- 4.  You can change the dummy data file location in the function `populatedb()` in `insert.py` 
    - e.g. if the location is `qp/MeasReport-cell-pdcp-b.csv`
        ```py
        def populatedb():
            df = pd.read_csv('qp/MeasReport-cell-pdcp-b.csv')
            df = time(df)
            db = INSERTDATA()
            db.client.write_points(df, 'QP')
        ```

## New training process designed by K. D.

Note: All the procedure below is difference from qp xapp
- 1. Take first difference until data is stationary 
- 2. Calculate the PACF to pick out lag that have strong correlation with the current situation
- 3. Determine the max lag considered
- 4. fit An AR model for a cell
- 5. performance test doesn't in here

### Make stationary
```py

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

      
      self.stationary_data = df.diff().dropna() # take the first difference 
      self.reverse.append(self.stationary_data) # store the reverse data, used for transforming to the orignal scale

      
      res_adf_processed = self.adfuller_test(self.stationary_data)
  
      self.diff =  self.diff + 1 

      # take the first difference until the data is stationary
      while not res_adf_processed:
        self.stationary_data = self.stationary_data.diff().dropna()
        self.reverse.append(self.stationary_data)
        self.diff = self.diff + 1

      self.diff_time[cell] = self.diff  # store the times we take difference for a cell
    else:
      self.stationary_data = df
      self.diff_time[cell] = self.diff
 ```


### Calculate the PACF to get the model lag candidate

```py

def pacf_calculate(self):
    """ Calculate the PACF
    """
    self.cell_lag = len(self.stationary_data) 
 
    self.pacf_list = pacf(self.stationary_data,nlags=self.cell_lag) # call pacf() to get every pacf of lag
  
    avg, dev = self.stationary_data.mean(), self.stationary_data.std()
    df_norm = (self.stationary_data - avg) / dev
    confidence_interval_array = st.norm.interval(alpha=0.95,
                  loc=np.mean(df_norm),
                  scale=st.sem(df_norm))  # calculate 0.95 confidence interval to know which lag have the strong correlation with the current situation
    self.confidence_interval = abs(confidence_interval_array[0][0])  
 ```



```py

   def ar_model_candidate(self):
    """ To pick out lag that have strong correlation with the current situation
    """
    ar_order = -1
    for i in self.pacf_list:
      ar_order = ar_order + 1
      if i!=1 and abs(i)>=self.confidence_interval: 
        self.ar_orders.append(ar_order) # if the order of lag have  the strong correlation with the current situation , store it as the model candidate
 ```

###  Determine the max lag considered
```py
  def find_optimal_lag(self):

    """ Determine the max lag considered
    """
    res = []
    if len(self.ar_orders) != 0 : 
      fitted_model_dict = {}

      # create the model 
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
        
          model_fit_lag_orders[ar_order] = fitted_model_dict[ar_order].bic
        temp = min(model_fit_lag_orders.values())
        res = [key for key in model_fit_lag_orders if model_fit_lag_orders[key] == temp]
      except:
        """
        TODO:  Can't build a few orders :　SVD did not converge  
        """  
        res.append(self.cell_lag) 
  
    else:
      """
      TODO:  what if we can't find an order in the 0.95 confidence interval ?
        --> does it means the training data isn't enough ?
      
      """
      res.append(self.cell_lag)
    return res[0]
```

### fit An AR model for a cell

```py
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
            model = ARIMA(md.stationary_data, order=(optimal_lag,0))  # do 4.
            model_fit = model.fit() 

        # if SVD did not converge  , use another best lag 
        except:
            for i in range(optimal_lag):
                optimal_lag = optimal_lag - 1
                try:
                    model = ARIMA(md.stationary_data, order=(optimal_lag,0))  # do 4. 
                    model_fit = model.fit()    
                    if model_fit is not None:
                        break
                except:
                    pass
```