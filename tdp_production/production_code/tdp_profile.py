from cnn_predictor import *
from plotTDP import *
from early_tdp import *
import json
import time

class TDP_Profile:
    def __init__(self, csv_path, model_config):
        self.csv_path = csv_path
        self.model_config = model_config

    def load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    
    def tdp_prediction(self, model_path, fig):
        load_start = time.time()
        predictor = CNN_Predictor(model_path)
        model = predictor.load_model()
        load_end = time.time()

        pred_start = time.time()
        result = predictor.predict(model, fig)
        pred_end = time.time()

        result["pred_time"] = pred_end - pred_start
        result["load_time"] = load_end - load_start
        return result
    
    def tdp_plot(self):
        df = pd.read_csv(self.csv_path)
        failure_head_list = get_bad_head(df=df)
        tdp = TDP(df=df, bad_head_list=failure_head_list)
        fig = tdp.display()
        return fig
    
    def check_other(self, result):
        if "other" not in result["class"]:
            return True
        else:
            return False
        
    ## flow: high/low -> aahr -> inc/dec -> hl_inc/dec -> recovery -> early tdp -> normal OD<ID
    def tdp_predict_profile(self):
        model_path_dict = self.load_json(self.model_config)

        fig = self.tdp_plot()
        plt.close(fig)

        proc_start = time.time()
        ## HIGH-LOW
        result = self.tdp_prediction(model_path_dict["model_high_low_path"], fig)
        if self.check_other(result):
            proc_end = time.time()
            result["proc_time"] = proc_end - proc_start
            return result

        ## AIRMIX-DELTA-HELIUM LEAK-REVERT
        result = self.tdp_prediction(model_path_dict["model_aahr_path"], fig)
        if self.check_other(result):
            proc_end = time.time()
            result["proc_time"] = proc_end - proc_start
            return result
        
        ## NORMAL INCREASE-DECREASE
        result = self.tdp_prediction(model_path_dict["model_increase_decrease_path"], fig)
        if self.check_other(result):
            proc_end = time.time()
            result["proc_time"] = proc_end - proc_start
            return result
        
        ## RECOVERY
        result = self.tdp_prediction(model_path_dict["model_recovery_path"], fig)
        if self.check_other(result):
            proc_end = time.time()
            result["proc_time"] = proc_end - proc_start
            return result
        
        ## HIGH DECREASE-LOW INCREASE
        result = self.tdp_prediction(model_path_dict["model_hl_inc_dec_path"], fig)
        if self.check_other(result):
            proc_end = time.time()
            result["proc_time"] = proc_end - proc_start
            return result
        
        ## EARLY TDP
        result = self.check_early_tdp(self.csv_path)
        result_tdp = result.get("TDP_early")
        if result_tdp == "yes":
            proc_time = time.time() - proc_start
            result = {
                "class": "early_tdp", 
                "confidence": np.nan, 
                "proc_time": proc_time
            }
            return result


    def check_early_tdp(self, csv_path):
        df = pd.read_csv(csv_path)
        bh = get_bad_head(df=df)
        early_tdp = EARLY_TDP(df=df, bh=bh)
        result = early_tdp.run()
        return result

