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

## flow: high/low -> aahr -> inc/dec -> hl_inc/dec -> recovery -> early tdp -> normal OD<ID

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
    
    def load_internal_model(self, model_path:str) -> tuple():
        predictor = CNN_Predictor(model_path)
        model = predictor.load_model()
        return predictor, model
    
    def tdp_prediction(self, model_path:str, fig:matplotlib.figure.Figure) -> dict():
        predictor, model = self.load_internal_model(model_path)
        result = predictor.predict(model, fig)
        return result
    
    ## main method to run
    def tdp_predict_profile(self, csv_path:str, prediction_mode="all") -> list():
        self.csv_path = csv_path
        self.prediction_mode = prediction_mode ## all, condition, multiprocess
        
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

            elif self.prediction_mode == "condition":
                result = self.prediction_flow_condition_v1(fig, model_path_dict, bad_head_list)
                
            result["head"] = bad_head_list
            results.append(result)

        final_result = self.post_process_return(results)
        return final_result ## return final result

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
        
        elif self.prediction_mode == "condition":
            temp_results = []
            new_results = {}
            
            for idx in range(len(results)):
                temp_result = self.re_ranking_cond([results[idx]])
                temp_results.append(temp_result)
                
            classes = []
            confidences = []
            early_flags = []
            heads = []

            for result in temp_results:
                class_ = result.get("class")
                confidence = result.get("confidence")
                early_flag = result.get("early_tdp_flag")
                head = result.get("head")

                classes.append(class_)
                confidences.append(confidence)
                early_flags.append(early_flag)
                heads.append(head)
            
            new_results["head"] = [head[0] for head in heads]
            new_results["class"] = classes
            new_results["confidence"] = [conf[0] for conf in confidences]
            new_results["early_tdp_flag"] = [flag[0] for flag in early_flags]

            return new_results
        
    def re_ranking_cond(self, results:list()) -> dict():        
        results = results[0]
        other_counter = 0
        all_keys = list(results.keys())
        n_all_keys = len(all_keys)

        other_flag = False
        other_dict = {}
        non_other_dict = {}
        temp_words = []

        for key, value in results.items():
            if "head" in key or "early_tdp_flag" in key:
                continue
            # all_output[key] = value
            if "other" in key:
                other_counter += 1
                other_dict[key] = value
            else:
                non_other_dict[key] = value

            if (other_counter == n_all_keys - 2): ## exclude head and early tdp flag
                other_flag = True
                break
            
        max_value_non_other = np.max(list(non_other_dict.values())) if len(non_other_dict) > 0 else 0
        avg_other_confi  = np.mean(list(other_dict.values()))
        print(f"other_flag: {other_flag}")
        print(f"avg_other_confi: {avg_other_confi}")
        print(f"max_value_non_other: {max_value_non_other}")
        
        if (other_flag or other_counter >= 4) and (avg_other_confi > 90.0 and avg_other_confi > max_value_non_other):
            return_dict = {}
            avg_other_confi  = np.mean(list(other_dict.values()))
            
            return_dict["head"] = results["head"]
            return_dict["class"] = ["other"]
            return_dict["confidence"] = [avg_other_confi]
            return_dict["early_tdp_flag"] = results["early_tdp_flag"]
        
        if not other_flag:
            return_dict = {}
            temp_list = []
            sorted_confidence = sorted(non_other_dict.values(), reverse=True)
            
            ## restore key to value
            for confi in sorted_confidence:
                for key, value in non_other_dict.items():
                    if confi == value:
                        temp_list.append(key)

            return_dict["head"] = results["head"]
            return_dict["class"] = temp_list
            return_dict["confidence"] = [sorted_confidence[0]]
            return_dict["early_tdp_flag"] = results["early_tdp_flag"]
        
        if len(non_other_dict) > 0:
            for words in return_dict["class"]:
                new_words = self.word_post_process(words)
                temp_words.append(new_words)
            return_dict["class"] = temp_words

        return return_dict
        
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
            result["class"][0] = "early_tdp"

        return result
    
    def prediction_flow_condition_v1(self, fig:matplotlib.figure.Figure, model_path_dict:dict(), bad_head_list:list()) -> dict():
        ## EARLY TDP
        early_flag = self.check_early_tdp(self.csv_path, bad_head_list).get("TDP_early")
        ## HIGH-LOW
        hi_lo_result = self.tdp_prediction(model_path_dict["model_high_low_path"], fig)
        ## AIRMIX-DELTA-HELIUM LEAK-REVERT
        aahr_result = self.tdp_prediction(model_path_dict["model_aahr_path"], fig)
        ## NORMAL INCREASE-DECREASE
        inc_dec_result = self.tdp_prediction(model_path_dict["model_increase_decrease_path"], fig)
        ## RECOVERY
        recov_result = self.tdp_prediction(model_path_dict["model_recovery_path"], fig)
        ## HIGH DECREASE-LOW INCREASE
        hl_inc_dec_result = self.tdp_prediction(model_path_dict["model_hl_inc_dec_path"], fig)
        ## NORMAL OD<ID AND ID>OD
        # normal_od_id_result = self.tdp_prediction(model_path_dict["model_normal_od_id_path"], fig)

        ## get first of prediction and confidence all model
        hi_lo_class, hi_lo_confidence = hi_lo_result.get("class")[0], hi_lo_result.get("confidence")[0]
        aahr_class, aahr_confidence = aahr_result.get("class")[0], aahr_result.get("confidence")[0]
        inc_dec_class, inc_dec_confidence = inc_dec_result.get("class")[0], inc_dec_result.get("confidence")[0]
        recov_class, recov_confidence = recov_result.get("class")[0], recov_result.get("confidence")[0]
        hl_inc_dec_class, hl_inc_dec_confidence = hl_inc_dec_result.get("class")[0], hl_inc_dec_result.get("confidence")[0]
        # normal_od_id_class, normal_od_id_confidence = normal_od_id_result.get("class")[0], normal_od_id_result.get("confidence")[0]

        # results = {
        #     "Hi-Low":[f"{self.word_post_process(hi_lo_class)}", hi_lo_confidence],
        #     "AAHR":[f"{self.word_post_process(aahr_class)}", aahr_confidence],
        #     "Normal-inc/dec":[f"{self.word_post_process(inc_dec_class)}", inc_dec_confidence],
        #     "Recovery":[f"{self.word_post_process(recov_class)}", recov_confidence],
        #     "Hi-Low-inc/dec":[f"{self.word_post_process(hl_inc_dec_class)}", hl_inc_dec_confidence],
        #     # f"{self.word_post_process(normal_od_id_class)}": normal_od_id_confidence,
        #     "early_tdp_flag": early_flag
        # }
        
        results = {
            f"{hi_lo_class}": hi_lo_confidence,
            f"{aahr_class}": aahr_confidence,
            f"{inc_dec_class}": inc_dec_confidence,
            f"{recov_class}": recov_confidence,
            f"{hl_inc_dec_class}":hl_inc_dec_confidence,
            # f"{self.word_post_process(normal_od_id_class)}": normal_od_id_confidence,
            "early_tdp_flag": early_flag
        }
        
        print(f"All results: {results}")
        return results    

    def check_early_tdp(self, csv_path:str, bad_head_list:list()=[]) -> dict():
        df = pd.read_csv(csv_path)
        early_tdp = EARLY_TDP(df=df, bad_head_list=bad_head_list)
        result = early_tdp.run()
        return result