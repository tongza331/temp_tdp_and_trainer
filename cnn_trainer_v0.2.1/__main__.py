from cnn_trainer import CNN_Trainer
from cnn_predictor import CNN_Predictor
from clip_trainer import CLIP_Trainer
import pandas as pd
import matplotlib.pyplot as plt
from plotTDP import *
import time
from natsort import natsorted

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

if __name__ == "__main__":
    # dataset_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\temp_classify"
    
    # trainer = CNN_Trainer(dataset_path)

    ######################### NORMAL TRAINING ##########################
    # params_v1 = {
    #     # "model_name": "volo_d1_224.sail_in1k",
    #     "model_name":"sequencer2d_m.in1k",
    #     "EPOCHS": 50,
    #     "SAVED": True,
    #     "lr": 1e-4,
    #     "weight_decay": 1e-3,
    #     "model_version": "all_v2_noNorm",
    #     "batch_size": 32,
    #     "valid_size":0.4,
    #     "test_size":0.4,
    #     "sched":"cosine",
    #     "opt":"adamw",
    #     "use_wandb":False,
    #     "CUSTOM_MODEL":False,
    # }
    # trainer.train_model_v1(**params_v1)
    # params_v2 = {
    #     "model_name": "sequencer2d_m.in1k",
    #     "EPOCHS": 50,
    #     "use_lookahead": True,
    #     "SAVED": True,
    #     "num_accumulate":5,
    #     "lr": 1e-4,
    #     "weight_decay": 1e-3,
    #     "model_version": "all_v2_Norm",
    #     "batch_size": 32,
    #     "valid_size":0.4,
    #     "test_size":0.4,
    #     "sched":"cosine",
    #     "opt":"adamw",
    #     "CUSTOM_MODEL":False,
    #     "use_accumulate":True,
    # }
    # trainer.train_model_v2(**params_v2)

    ########################## CROSS VALIDATION ##########################
    # params_validate = {
    #     "model_name": "sequencer2d_s.in1k",
    #     "num_epochs": 20,
    #     "train_batch_size": 32,
    #     "eval_batch_size": 64,
    #     "k_splits": 6,
    #     # "num_accumulate": 5,
    #     "lr":1e-4,
    #     "weight_decay":1e-3,
    #     "model_version": "all_noNorm_kv1",
    # }
    # trainer.train_cross_validation_v1(**params_validate)
    
    ########################## CLIP-Logistic Regression ##########################
    # dataset_path2 = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease\high_low_other"
    # clip_trainer = CLIP_Trainer(dataset_path)
    # clip_trainer.train_model(model_method="logistic")

    ########################## PREDICTION ##########################
    # model_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\models\models_cuda\best_sequencer2d_s.in1k_all_noNorm_kv1_cuda_fold3.pt"
    # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease_subplot\low_tdp_2\8LK0U1LX_0.png"
    # folder_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease_subplot\low_tdp_2"
    # predictor = CNN_Predictor(model_path)
    # model = predictor.load_model()
    # result = predictor.predict(model, image_path)
    # print(f"Class prediction: {result['class']} | Confidence: {result['confidence']}")
    # predictor.predict_folder(model, folder_path)    
    
    ##### MODEL ENSEMBLE #####
    model_path1 = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\models\models_cuda\best_sequencer2d_s.in1k_all_noNorm_kv1_cuda_fold3.pt"
    model_path2 = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\models\last_resnet34d_all_v3_noNorm.pt"
    model_path4 = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\models\best_volo_d1_224.sail_in1k_all_v1_noNorm.pt"
    # model_path3 = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\models\best_sequencer2d_m.in1k_all_v1_noNorm.pt"
    
    # ##### TESTING WEIGHT ENSEMBLE WITH FOLDER #####
    # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease_subplot\low_tdp_2\8LK0U1LX_0.png"
    # folder_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\temp_classify\airmix_helium_leak_full"
    # predictor = CNN_Predictor(model_path=None)
    # model_list = [model_path1, model_path2, model_path4]
    
    # for file in natsorted(os.listdir(folder_path)):
    #     result = predictor.predict_weight_ensemble(model_list, os.path.join(folder_path, file), is_path=True)
    #     print(result)
    
    ##### TESTING WEIGHT ENSEMBLE #####
    csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\VL_TDP\6FPW\TDP_6FPW.csv"
    # csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\VL_TDP\6FPR\TDP_6FPR.csv"
    # csv_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\VL_TDP\6FPV\TDP_6FPV_R_W.csv"
    # hddsn = "2GHYLKTW"
    # csv_path = f"C:/Users/1000303969\OneDrive - Western Digital/work/tdp classification/VL_TDP/6FPV_v2/TDP_{hddsn}.csv"
    
    predictor = CNN_Predictor(model_path=None)
    model_dict = [model_path2, model_path1, model_path4]
    
    split_ec = True

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

    for hddsn in hddsn_list:
        df_filter = df[df["hddsn"] == hddsn]
        failure_head_list = get_bad_head(df=df_filter)
        n_fh = len(failure_head_list)
        for fh_i in range(n_fh):
            tdp = TDP(df=df_filter, bad_head_list=[failure_head_list[fh_i]])
            fig = tdp.display()
            plt.close(fig)
            start = time.time()
            fig = predictor.convert_to_arr(fig)
            result = predictor.predict_weight_ensemble(model_dict, fig, is_path=False)
            end = time.time()
            print(hddsn, failure_head_list[fh_i], f"Time: {end-start}")
            print(result)
            print("==================================================")
