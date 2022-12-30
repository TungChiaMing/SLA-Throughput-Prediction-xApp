# ==================================================================================
#       Copyright (c) 2020 AT&T Intellectual Property.
#       Copyright (c) 2020 HCL Technologies Limited.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ==================================================================================
"""
qp module main -- using Time series ML predictor

RMR Messages:
 #define TS_UE_LIST 30000
 #define TS_QOE_PREDICTION 30002
30000 is the message type QP receives from the TS;
sends out type 30002 which should be routed to TS.

"""
"""Ken QP xApp ML Enhanced
    import pandas as pd
"""
import insert
import os
import json
from mdclogpy import Logger
from ricxappframe.xapp_frame import RMRXapp, rmr
from prediction import forecast
from qptrain import train
from database import DATABASE, DUMMY
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

"""Ken QP xApp ML Enhanced
    training_data_len_dict = {}
"""
training_data_len_dict = {}
# pylint: disable=invalid-name
qp_xapp = None
Do_Prediction_One_time = False
Do_Prediction_One_time_cnt = 0
logger = Logger(name=__name__)


def post_init(self):
    """
    Function that runs when xapp initialization is complete
    """
    self.predict_requests = 0
    logger.debug("QP xApp started")


def qp_default_handler(self, summary, sbuf):
    """
    Function that processes messages for which no handler is defined
    """
    logger.debug("default handler received message type {}".format(summary[rmr.RMR_MS_MSG_TYPE]))
    # we don't use rts here; free this
    self.rmr_free(sbuf)


def qp_predict_handler(self, summary, sbuf):
    """
    Function that processes messages for type 30000
    """
    logger.debug("predict handler received payload {}".format(summary[rmr.RMR_MS_PAYLOAD]))
    
   
    pred_msg = predict(summary[rmr.RMR_MS_PAYLOAD])
    self.predict_requests += 1
    # we don't use rts here; free this
    self.rmr_free(sbuf)
    success = self.rmr_send(pred_msg.encode(), 30002)
    logger.debug("Sending message to ts : {}".format(pred_msg))  # For debug purpose
    if success:
        logger.debug("predict handler: sent message successfully")
    else:
        logger.warning("predict handler: failed to send message")


def cells(ue):
    """
        Extract neighbor cell id for a given UE
    """
    db.read_data(meas='liveUE', limit=1, ueid=ue)
    df = db.data

    nbc = df.filter(regex='nbCell').values[0].tolist()
    srvc = df.filter(regex='nrCell').values[0].tolist()
    return srvc+nbc

"""Ken QP xApp ML Enhanced
    db.read_data(meas='QP')
    cell_list = db.data['nrCellIdentity'].drop_duplicates()
    global training_data_len_dict

"""
def predict(payload):
    """
     Function that forecast the time series
    """
    tp = {}
    global training_data_len_dict

    global Do_Prediction_One_time
    global Do_Prediction_One_time_cnt
    payload = json.loads(payload)
    
    ueid = payload['UEPredictionSet'][0]

    cell_list = cells(ueid)
    db.read_data(meas='QP')
    cell_list = db.data['nrCellIdentity'].drop_duplicates()

    for cid in cell_list:
        mcid = cid.replace('/', '')
        db.read_data(meas='QP', cellid=cid, limit=11)
        if len(db.data) != 0:
            
            """Ken QP xApp ML Enhanced
                training_data_len_dict = {}
                training_data_len = train(db, cid)
                training_data_len_dict[cid] = training_data_len
                db.read_data(meas='QP', cellid=cid, limit=training_data_len_dict[cid])
                inp = db.data
                nobs = 3 
            """
            if not os.path.isfile('qp/' + mcid):
                training_data_len = train(db, cid)
                training_data_len_dict[mcid] = training_data_len
            db.read_data(meas='QP', cellid=cid, limit=training_data_len_dict[mcid])
            inp = db.data.copy()
         
            df_f = forecast(inp, mcid, 3)
            pdcpBytesDl =  inp[['nrCellIdentity','pdcpBytesDl']]
            if df_f is not None:
                tp[cid] = df_f.values.tolist()[0]
                df_f['cellid'] = cid
                df_pred =  df_f.rename(columns = {"pdcpBytesDl":"pdcpBytesDl_predict"})
                df_pred = df_pred['pdcpBytesDl_predict'] 
                df_pred_upload = pd.concat([pdcpBytesDl,df_pred],axis=1)
                if (Do_Prediction_One_time_cnt >= len(cell_list)):
                    Do_Prediction_One_time = True
                else:
                    Do_Prediction_One_time_cnt = Do_Prediction_One_time_cnt + 1
                if(Do_Prediction_One_time == False):
                    db.write_prediction(df_pred_upload, 'QP_Prediction')
                
                
            else:
                tp[cid] = [None, None]
    return json.dumps({ueid: tp})


def start(thread=False):
    """
    This is a convenience function that allows this xapp to run in Docker
    for "real" (no thread, real SDL), but also easily modified for unit testing
    (e.g., use_fake_sdl). The defaults for this function are for the Dockerized xapp.
    """
    logger.debug("QP xApp starting")
    global qp_xapp
    global db
    if not thread:
        insert.populatedb()   # temporory method to popuate db, it will be removed when data will be coming through KPIMON to influxDB
        db = DATABASE('UEData')
    else:
        db = DUMMY()
    fake_sdl = os.environ.get("USE_FAKE_SDL", None)
    qp_xapp = RMRXapp(qp_default_handler, rmr_port=4560, post_init=post_init, use_fake_sdl=bool(fake_sdl))
    qp_xapp.register_callback(qp_predict_handler, 30000)
    qp_xapp.run(thread)


def stop():
    """
    can only be called if thread=True when started
    TODO: could we register a signal handler for Docker SIGTERM that calls this?
    """
    global qp_xapp
    qp_xapp.stop()


def get_stats():
    """
    hacky for now, will evolve
    """
    global qp_xapp
    return {"PredictRequests": qp_xapp.predict_requests}
