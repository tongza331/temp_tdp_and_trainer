from tdp_profile import TDP_Profile

if __name__ in "__main__":
    csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\VL_TDP\6FPW\3ZH607HZ.csv"
    # csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\prep_data\2GHM176W_TDL6_16.csv"
    model_config = "code/tdp_production/production_code/model_path.json"
    tdp_profile = TDP_Profile(csv_path, model_config, prediction_mode="all")
    result = tdp_profile.tdp_predict_profile()
    print(result)