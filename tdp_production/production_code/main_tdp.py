from tdp_profile import *

if __name__ in "__main__":
    csv_path = "code/tdp_production/production_code/2GHKMN7S_TDMO_19.csv"
    model_config = "code/tdp_production/production_code/model_path.json"
    tdp_profile = TDP_Profile(csv_path, model_config)
    result = tdp_profile.tdp_predict_profile()
    print(result)