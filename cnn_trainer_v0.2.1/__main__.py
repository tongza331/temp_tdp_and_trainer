from cnn_trainer import CNN_Trainer
from cnn_predictor import CNN_Predictor
from clip_trainer import CLIP_Trainer

if __name__ == "__main__":
    dataset_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\temp_classify"
    
    trainer = CNN_Trainer(dataset_path)

    ######################### NORMAL TRAINING ##########################
    params_v1 = {
        "model_name": "volo_d1_224.sail_in1k",
        # "model_name":"sequencer2d_l.in1k",
        "EPOCHS": 50,
        "SAVED": True,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "model_version": "all_v1_noNorm",
        "batch_size": 32,
        "valid_size":0.4,
        "test_size":0.4,
        "sched":"cosine",
        "opt":"adamw",
        "use_wandb":False,
        "CUSTOM_MODEL":False,
    }
    trainer.train_model_v1(**params_v1)
    # params_v2 = {
    #     "model_name": "resnet50d",
    #     "EPOCHS": 50,
    #     "use_lookahead": True,
    #     "SAVED": False,
    #     "num_accumulate":4,
    #     "lr": 1e-3,
    #     "weight_decay": 1e-4,
    #     "model_version": "all_v1",
    #     "batch_size": 32,
    #     "valid_size":0.5,
    #     "test_size":0.4,
    #     "sched":"cosine",
    #     "opt":"adamw",
    #     "use_wandb":False,
    #     "CUSTOM_MODEL":False,
    #     "use_accumulate":True,
    # }
    # trainer.train_model_v2(**params_v2)

    ########################## CROSS VALIDATION ##########################
    # params_validate = {
    #     "model_name": "resnet18d",
    #     "num_epochs": 20,
    #     "train_batch_size": 32,
    #     "eval_batch_size": 64,
    #     "k_splits": 6,
    #     "num_accumulate": 5,
    #     "model_version": "all_noNorm_kv1",
    # }
    # trainer.train_cross_validation(**params_validate)

    ########################## PREDICTION ##########################
    # model_path = r"C:\Users\1000303969\Downloads\best_resnet50d_fold_5.pt"
    # model_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\models\best_resnet50d_aahr_kv5_fold7.pt"
    # model_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\code\tdp_production\production_models\best_resnet34d_increase_decrease_v1_2_best.pt"
    # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\all_sample\2HGGAG0N_0.png" 
    # # # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\temp_classify\helium_leak\2GH771MS_0.png"
    # folder_path = r"C:\Users\1000303969\Downloads\helium_leak"

    # # # # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease_subplot\delta_tdp\2GHMY6XS_0.png"
    
    # predictor = CNN_Predictor(model_path)

    # model = predictor.load_model()
    # result = predictor.predict(model, image_path)
    # print(f"Class prediction: {result['class']} | Confidence: {result['confidence']}")

    # predictor.predict_folder(model, folder_path)    

    ########################## CLIP-Logistic Regression ##########################
    # dataset_path2 = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease\high_low_other"
    # clip_trainer = CLIP_Trainer(dataset_path)
    # clip_trainer.train_model(model_method="logistic")