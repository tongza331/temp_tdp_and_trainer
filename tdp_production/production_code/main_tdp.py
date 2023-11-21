from tdp_profile import TDP_Profile
import pandas as pd
import os
import time

model_config = "code/tdp_production/production_code/model_path.json"

W_list = [
    "2TGLXRTZ",
    "3ZH607HZ",
    "3ZH61Z0Z",
    "3ZH609ZZ",
    "2TGLYS7Z",
    "3ZH61PWZ",
    "3ZH5Z5XZ",
    "3ZH620SZ",
    "3ZH6200Z",
    "3ZH5X4UZ",
    "3ZH5LL0Z",
    "2TGLXTRZ"
]

R_list = [
    "2GJ5BMAD",
    "2TGLU8RZ"
]

V_list = [
    "2GG02VGL",
    "2GH2KP7S",
    "2GHR65SS",
    "2GJ2XK9S",
    "2GJ3R6SD",
    "2GJ68Y3S",
    "2TGLJDWZ",
    "2GJ69HJS",
    "3ZH4NYAZ",
    "ZGG0AABA",
]

if __name__ in "__main__":
    # csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\VL_TDP\6FPW\TDP_6FPW.csv"
    # csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\VL_TDP\6FPR\TDP_6FPR.csv"
    # csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\VL_TDP\6FPV\TDP_6FPV_R_W.csv"
    hddsn = "2HGEW5XN"
    csv_path = f"C:/Users/1000303969/OneDrive - Western Digital/work/tdp classification/VL_TDP/6FPV_v2/TDP_{hddsn}.csv"
    
    split_ec = False

    df = pd.read_csv(csv_path)
    ec = df["pfcode"].unique()
    print(ec)

    if split_ec:
        if ("6FPR" in ec) and ("6FPV" not in ec):
            hddsn_list = R_list
        elif "6FPW" in ec:
            hddsn_list = W_list
        else:
            hddsn_list = V_list
    else:
        print("not split ec")
        hddsn_list = df["hddsn"].unique()

    tdp_profile = TDP_Profile(model_config)
    
    for hddsn in hddsn_list:
        df_filter = df[df["hddsn"] == hddsn]
        try:
            ## save df to csv
            base_dir = os.path.dirname(csv_path)
            df_filter.to_csv(os.path.join(base_dir, hddsn+".csv"), index=False)
            ## load csv
            csv_path = os.path.join(base_dir, hddsn+".csv")
            start = time.time()
            result = tdp_profile.tdp_predict_profile(csv_path, prediction_mode="ensemble")
            end = time.time()
            print(hddsn, result, end-start)
            print("==================================================")
        except Exception as e:
            print(hddsn, "error", e)
            print("==================================================")
    
    # tdp_profile = TDP_Profile(csv_path, model_config, prediction_mode="condition") ## all, condition
    # result = tdp_profile.tdp_predict_profile()
    # print(result)