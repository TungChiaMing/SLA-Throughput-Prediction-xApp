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
# import pandas as pd
import os
import joblib
import pandas as pd
from qptrain import PROCESS

"""Ken QP xApp ML Enhanced"""
def forecast(data, cid, nobs=10):
    """
     forecast the time series using the saved model.
    """
    Training_Not_Predictable = False
    ps = PROCESS(data.copy())
    
    predictable = ps.process(cid)
    file_name = 'qp/'+cid.replace('/', '')
    if predictable:
        
        if os.path.isfile(file_name):

            model = joblib.load(file_name)
            pred = model.forecast(steps=nobs)
        else:
            pred = []
            for i in range(nobs):
                pred.append(ps.data['pdcpBytesDl'][0]) # append timestamp
                df_f = pd.DataFrame(pred, columns=ps.data.columns)
                df_f.index = pd.date_range(start=ps.data.index[-1], freq='10ms', periods=len(df_f))
                Training_Not_Predictable = True
         

    # if it's not predictable, just remain the same value
    else:
        pred = []
        for i in range(nobs):
            pred.append(ps.data['pdcpBytesDl'][0])
            df_f = pd.DataFrame(pred, columns=ps.data.columns)
            df_f.index = pd.date_range(start=ps.data.index[-1], freq='10ms', periods=len(df_f))
            Training_Not_Predictable = True
    if Training_Not_Predictable is True:
        pass
    elif pred is not None:
        df_f = pd.DataFrame(pred[0], columns=ps.data.columns)
        df_f = ps.invert_transformation(ps.data, df_f, cid)
    else:
        pass
    df_f = df_f[ps.data.columns].astype(int)

    return df_f
