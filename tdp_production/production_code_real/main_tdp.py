from tdp_profile import TDP_Profile
from presto_connection import * ## to connect with BDP and Presto
import pandas as pd
import os
import time

model_config = ""
config_path = ""

## Simple USED 
### input as csv file
# if __name__ == "__main__":
#     csv_path = ""
#     profile = TDP_Profile(model_config=model_config)
#     results = profile.tdp_predict_profile(csv_path=csv_path, prediction_mode="ensemble")
#     print(results)

## Connect to PIPELINE
if __name__ == "__main__":
    
    serial_list = """
        
    """
    product = ""
    procid = ""
    pf_code = ""
    start_enddt = "" ## YYYYMMDD
    end_enddt = "" ## YYYYMMDD
    serial_list = serial_list.split()
    
    ## setting
    presto_connection = PrestoConnection(config_path, SAVE_CSV=False)
    return_df = pd.DataFrame()
    
    for hddsn in serial_list:
        MFGparams = {
            "serial":hddsn,
            "product":product,
            "procid":procid,
            "pf_code":pf_code,
            "start_enddt":start_enddt,
            "end_enddt":end_enddt,
        }
        mfg_df_dict = presto_connection.query(MFGparams, pfcode_mode=None)
        mfg_df = mfg_df_dict["MFG"]
        TDPparams = {
                "product": mfg_df["product"].unique()[0],
                "serial": mfg_df["hddsn"].unique()[0],
                "procid": mfg_df["procid"].unique()[0],
                "pf_code": mfg_df["pfcode"].unique()[0],
                "start_enddt": mfg_df["startdate"].unique()[0],
                "enddt": mfg_df["enddt"].unique()[0],
                "enddt64": mfg_df["enddt64"].unique()[0],
                "enddt66": mfg_df["enddt66"].unique()[0],
                "enddt68": mfg_df["enddt68"].unique()[0],
                "enddt90": mfg_df["enddt90"].unique()[0],
            }
        tdp_df = presto_connection.TDP_QUERY(**TDPparams)
        merge_data = MergeData(mfg_df, tdp_df)
        tdp_final_df = merge_data.merge_subcode()
        return_df = pd.concat([return_df, tdp_final_df])

    ## predict profile
    hddsn_list = return_df["hddsn"].unique()
    print("=================== PREDICT PROFILE ===================")
    profile = TDP_Profile(model_config=model_config)
    for hddn in hddsn_list:
        df = return_df[return_df["hddsn"] == hddn]
        results = profile.tdp_predict_profile(df=df, prediction_mode="ensemble") ## use input argument df instead of csv_path
        print(results)
