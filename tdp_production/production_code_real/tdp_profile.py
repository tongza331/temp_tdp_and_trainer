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

"""
The `TDP_Profile` class is responsible for loading a JSON file, processing word labels, loading internal models, making TDP predictions, and running the main TDP prediction profile.

Example Usage:
    # Create an instance of TDP_Profile
    profile = TDP_Profile(model_config="model_config.json")

    # Load a JSON file
    data = profile.load_json(path="data.json")

    # Process a word label
    processed_word = profile.word_post_process(word="other")

    # Load an internal model
    predictor, model = profile.load_internal_model(model_path="model.h5")

    # Make a TDP prediction
    result = profile.tdp_prediction(model_path="model.h5", fig=figure)

    # Run the TDP prediction profile
    final_result = profile.tdp_predict_profile(csv_path="data.csv", prediction_mode="ensemble")

Methods:
    - __init__(self, model_config:str): Initializes the TDP_Profile instance with a model configuration file path.
    - load_json(self, path:str) -> dict(): Loads a JSON file and returns its data as a dictionary.
    - word_post_process(self, word:str) -> str(): Processes a word label and returns the processed label.
    - load_internal_model(self, model_path:str="") -> tuple(): Loads an internal model and returns the predictor and model.
    - tdp_prediction(self, model_path:str, fig:matplotlib.figure.Figure) -> dict(): Makes a TDP prediction using the specified model and figure, and returns the prediction result.
    - tdp_predict_profile(self, csv_path:str, prediction_mode="ensemble") -> list(): Runs the main TDP prediction profile using the specified CSV file and prediction mode, and returns the final result.
    - prediction_all(self, fig:matplotlib.figure.Figure, model_path_dict:dict(), bad_head_list:list()) -> dict(): Makes TDP predictions for all models and returns the result.
    - prediction_weight_ensemble(self, fig:matplotlib.figure.Figure, model_path_dict:dict(), bad_head_list:list()) -> dict(): Makes TDP predictions using weighted ensemble models and returns the result.
    - post_process_return(self, results:list()) -> dict(): Post-processes the TDP prediction results and returns the final result.
    - check_early_tdp(self, csv_path:str, bad_head_list:list()=[]) -> dict(): Checks for early TDP and returns the result.

Fields:
    - model_config: The path to the model configuration file.
    - csv_path: The path to the CSV file.
    - prediction_mode: The prediction mode ("all" or "ensemble").
    - model_path_dict: A dictionary containing the paths to the internal models.
    - df: The DataFrame loaded from the CSV file.
    - failure_head_list: A list of bad head values.
    - results: A list to store the TDP prediction results.
    - early_result: The result of checking for early TDP.
"""
class TDP_Profile:
    def __init__(self, model_config:str):
        self.model_config = model_config

    def load_json(self, path:str) -> dict():
        with open(path, "r") as f:
            data = json.load(f)
        return data
    
    def word_post_process(self, word: str) -> str:
        if "other" in word:
            return "Other"
        if "airmix_helium_leak" in word:
            return "Airmix/Helium Leak"
        if "normal_ID-OD" in word:
            return "normal TDP and ID<OD"
        if "normal_OD-ID" in word:
            return "normal TDP and OD<ID"
        if "high_od+id" in word:
            return "high TDP and OD-ID"
        if "assembly" in word or "delta_od_id" in word:
            return "Delta OD-ID"
        if "high_tdp" in word:
            return "High TDP"
        if "low_tdp" in word:
            return "Low TDP"
        if "high_recovery" in word:
            return "High TDP and Recovery"
        if "low_recovery" in word:
            return "Low TDP and Recovery"
        if "high_decrease" in word:
            return "High TDP and Decrease"
        if "low_increase" in word:
            return "Low TDP and Increase"
        if "normal_increase" in word:
            return "Normal TDP and Increase"
        if "normal_decrease" in word:
            return "Normal TDP and Decrease"
        if "revert" in word:
            return "Reverse TDP"
        return word
    
    def load_internal_model(self, model_path:str="") -> tuple:
        if self.prediction_mode == "ensemble":
            predictor = CNN_Predictor(model_path=None)
            print("Loading internal model successfully")
            return predictor
        else:
            predictor = CNN_Predictor(model_path)
            model = predictor.load_model()
            return predictor, model

    def tdp_prediction(self, model_path: str, fig: matplotlib.figure.Figure) -> dict:
        
        if self.prediction_mode == "all":
            result = predictor.predict(model, fig)
            predictor, model = self.load_internal_model(model_path)
        elif self.prediction_mode == "ensemble":
            model_list = model_path
            predictor = self.load_internal_model(model_list)
            result = predictor.predict_weight_ensemble(model_list, fig)
        
        return result
    
    ## main method to run
    def tdp_predict_profile(self, csv_path:str=None, df:pd.DataFrame=None, prediction_mode="ensemble") -> list():
        self.csv_path = csv_path
        self.df = df
        self.prediction_mode = prediction_mode ## all, ensemble
        
        model_path_dict = self.load_json(self.model_config)
        
        if self.df is None and self.csv_path is not None:
            self.df = pd.read_csv(self.csv_path)

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
            
            result["hddsn"] = tdp.hddsn
            result["head"] = bad_head_list
            results.append(result)

        final_result = self.post_process_return(results)
        return final_result ## return final result

    def prediction_all(self, fig:matplotlib.figure.Figure, model_path_dict:dict(), bad_head_list:list()) -> dict():
        result = self.tdp_prediction(model_path_dict["model_all_path"], fig)
        
        ## check early tdp flag
        if self.csv_path is not None:
            early_input = self.csv_path
        else:
            early_input = self.df
            
        early_result = self.check_early_tdp(early_input, bad_head_list)
        result["early_tdp_flag"] = early_result.get("TDP_early")

        ## get predict class from top-k
        pred_class = result.get("class")

        ## word post process
        for i in range(len(pred_class)):
            result["class"][i] = self.word_post_process(pred_class[i])

        ## if pred is other and has early tdp flag, change word into "early_tdp"
        pred_class_top1 = result["class"][0]
        if pred_class_top1 == "other" and result["early_tdp_flag"][0] == "yes":
            result["class"][0] = "Early TDP"

        return result
    
    def prediction_weight_ensemble(self, fig:matplotlib.figure.Figure, model_path_dict:dict(), bad_head_list:list()) -> dict():
        result = self.tdp_prediction(model_path_dict["model_ensemble_path"], fig)
        
        ## check early tdp flag
        if self.csv_path is not None:
            early_input = self.csv_path
        else:
            early_input = self.df
            
        early_result = self.check_early_tdp(early_input, bad_head_list)
        result["early_tdp_flag"] = early_result.get("TDP_early")
        
        # get predict class from top-k
        pred_class = result.get("class")
        
        ## word post process
        for i in range(len(pred_class)):
            result["class"][i] = self.word_post_process(pred_class[i])
        
        ## if pred is other and has early tdp flag, change word into "ealry_tdp"
        pred_class_top1 = result["class"][0]
        if pred_class_top1 == "Other" and result["early_tdp_flag"][0] == "yes":
            print("EEEEEEEEEEEEEEEEEEEEEEEERLY!!!")
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
            
        
    def check_early_tdp(self, csv_path, bad_head_list:list()=[]) -> dict():
        if isinstance(csv_path, str):
            df = pd.read_csv(csv_path)
        else:
            df = csv_path ## input as pd
            
        early_tdp = EARLY_TDP(df=df, bad_head_list=bad_head_list)
        result = early_tdp.run()
        return result