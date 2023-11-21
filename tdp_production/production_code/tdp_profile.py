from cnn_predictor import *
from plotTDP import *
from early_tdp import *
from multiprocessing import Pool
import json
import time
import matplotlib
matplotlib.use('Agg')

#   {'airmix_helium_leak_full': 0,
#  'delta_od_id': 1,
#  'high_decrease': 2,
#  'high_od+id': 3,
#  'high_recovery': 4,
#  'high_tdp': 5,
#  'low_increase': 6,
#  'low_recovery': 7,
#  'low_tdp': 8,
#  'normal_ID-OD': 9,
#  'normal_OD-ID': 10,
#  'normal_decrease': 11,
#  'normal_increase': 12,
#  'other_all': 13,
#  'revert': 14}

class TDP_Profile:
    def __init__(self, model_config:str):
        self.model_config = model_config

    def load_json(self, path:str) -> dict():
        with open(path, "r") as f:
            data = json.load(f)
        return data
    
    def word_post_process(self, word:str) -> str():
        if "other" in word:
            return "other"
        elif "airmix_helium_leak" in word:
            return "Airmix and Helium Leak"
        elif "normal_ID-OD" in word:
            return "normal_ID<OD"
        elif "normal_OD-ID" in word:
            return "normal_OD<ID"
        elif "high_od+id" in word:
            return "high_OD>ID"
        elif "assembly" in word or "delta_od_id" in word:
            return "Delta OD < ID"
        elif "high_tdp" in word:
            return "High TDP"
        elif "low_tdp" in word:
            return "Low TDP"
        elif "high_recovery" in word:
            return "High TDP and Recovery"
        elif "low_recovery" in word:
            return "Low TDP and Recovery"
        elif "high_decrease" in word:
            return "High TDP and Decrease"
        elif "low_increase" in word:
            return "Low TDP and Increase"
        elif "normal_increase" in word:
            return "Normal TDP and Increase"
        elif "normal_decrease" in word:
            return "Normal TDP and Decrease"
        elif "revert" in word:
            return "Revert TDP"
        else:
            return word
    
    def load_internal_model(self, model_path:str="") -> tuple():
        if self.prediction_mode == "ensemble":
            predictor = CNN_Predictor(model_path=None)
            return predictor
        else:
            predictor = CNN_Predictor(model_path)
            model = predictor.load_model()
            return predictor, model

    def tdp_prediction(self, model_path:str, fig:matplotlib.figure.Figure) -> dict():
        if self.prediction_mode == "all":
            predictor, model = self.load_internal_model(model_path)
            result = predictor.predict(model, fig)
        elif self.prediction_mode == "ensemble":
            model_list = model_path
            predictor = self.load_internal_model()
            result = predictor.predict_weight_ensemble(model_list, fig)
        return result
    
    ## main method to run
    def tdp_predict_profile(self, csv_path:str, prediction_mode="ensemble") -> list():
        self.csv_path = csv_path
        self.prediction_mode = prediction_mode ## all, ensemble
        # print(f"Prediction mode: {self.prediction_mode}")
        
        model_path_dict = self.load_json(self.model_config)
        df = pd.read_csv(self.csv_path)

        failure_head_list = get_bad_head(df=df)
        n_fh = len(failure_head_list)
        results = []

        for fh_idx in range(n_fh):
            bad_head_list = [failure_head_list[fh_idx]]
            tdp = TDP(df=df, bad_head_list=bad_head_list)
            fig = tdp.display()
            plt.close(fig)

            if self.prediction_mode == "all":
                result = self.prediction_all(fig, model_path_dict, bad_head_list)

            elif self.prediction_mode == "ensemble":
                result = self.prediction_weight_ensemble(fig, model_path_dict, bad_head_list)
                
            result["head"] = bad_head_list
            results.append(result)

        final_result = self.post_process_return(results)
        return final_result ## return final result

    def prediction_all(self, fig:matplotlib.figure.Figure, model_path_dict:dict(), bad_head_list:list()) -> dict():
        result = self.tdp_prediction(model_path_dict["model_all_path"], fig)
        
        ## check early tdp flag
        early_result = self.check_early_tdp(self.csv_path, bad_head_list)
        result["early_tdp_flag"] = early_result.get("TDP_early")

        ## get predict class from top-k
        pred_class = result.get("class")

        ## word post process
        for i in range(len(pred_class)):
            result["class"][i] = self.word_post_process(pred_class[i])

        ## if pred is other and has early tdp flag, change word into "ealry_tdp"
        pred_class_top1 = result["class"][0]
        if pred_class_top1 == "other" and result["early_tdp_flag"][0] == "yes":
            result["class"][0] = "Early TDP"

        return result
    
    def prediction_weight_ensemble(self, fig:matplotlib.figure.Figure, model_path_dict:dict(), bad_head_list:list()) -> dict():
        result = self.tdp_prediction(model_path_dict["model_ensemble_path"], fig)
        
        ## check early tdp flag
        early_result = self.check_early_tdp(self.csv_path, bad_head_list)
        result["early_tdp_flag"] = early_result.get("TDP_early")
        
        # get predict class from top-k
        pred_class = result.get("class")
        
        ## word post process
        for i in range(len(pred_class)):
            result["class"][i] = self.word_post_process(pred_class[i])
        
        ## if pred is other and has early tdp flag, change word into "ealry_tdp"
        pred_class_top1 = result["class"][0]
        if pred_class_top1 == "other" and result["early_tdp_flag"][0] == "yes":
            result["class"][0] = "Early TDP"
            
        return result

    def post_process_return(self, results:list()) -> dict():
        if self.prediction_mode == "all":
            new_results = {}
            classes = []
            confidences = []
            early_flags = []
            heads = []

            for result in results:
                class_ = result.get("class")
                confidence = result.get("confidence")[0]
                early_flag = result.get("early_tdp_flag")[0]
                head = result.get("head")

                classes.append(class_)
                confidences.append(confidence)
                early_flags.append(early_flag)
                heads.append(head)
            
            new_results["head"] = [head[0] for head in heads]
            new_results["class"] = classes
            new_results["confidence"] = confidences
            new_results["early_tdp_flag"] = early_flags

            return new_results
        else:
            return results
            
        
    def check_early_tdp(self, csv_path:str, bad_head_list:list()=[]) -> dict():
        df = pd.read_csv(csv_path)
        early_tdp = EARLY_TDP(df=df, bad_head_list=bad_head_list)
        result = early_tdp.run()
        return result