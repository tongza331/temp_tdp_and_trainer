from cnn_trainer import CNN_Trainer
from cnn_predictor import CNN_Predictor
from clip_trainer import CLIP_Trainer

if __name__ == "__main__":
    dataset_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\temp_classify"
    
    trainer = CNN_Trainer(dataset_path)

    ######################### NORMAL TRAINING ##########################
    params_v2 = {
        "model_name": "resnet50d",
        "EPOCHS": 25,
        "use_lookahead": True,
        "SAVED": True,
        "num_accumulate":5,
        "lr": 1e-3,
        "weight_decay": 0.95,
        "model_version": "recovery_v2",
        "batch_size": 32,
        "valid_size":0.4,
        "test_size":0.4,
        "sched":"cosine",
        "opt":"adam",
        "use_wandb":False,
        "CUSTOM_MODEL":False,
        "use_accumulate":True,
    }
    trainer.train_model_v2(**params_v2)

    ######################### CROSS VALIDATION ##########################
    # params_validate = {
    #     "model_name": "resnet34d",
    #     "num_epochs": 20,
    #     "train_batch_size": 32,
    #     "eval_batch_size": 64,
    #     "k_splits": 6,
    #     "num_accumulate": 5,
    #     "model_version": "recovery_kv1",
    # }
    # trainer.train_cross_validation(**params_validate)

    ########################## PREDICTION ##########################
    # model_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\models\best_resnet34d_recovery_kv1_fold4.pt"

    # # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\all_sample\2GHV6DJS_0.png" 
    # # # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\temp_classify\helium_leak\2GH771MS_0.png"
    # folder_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\temp_classify\other_recovery"

    # # # # image_path = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease_subplot\delta_tdp\2GHMY6XS_0.png"
    
    # predictor = CNN_Predictor(model_path)

    # model = predictor.load_model()
    # # result = predictor.predict(model, image_path)
    # # print(f"Class prediction: {result['class']} | Confidence: {result['confidence']}")

    # predictor.predict_folder(model, folder_path)    

    ########################## CLIP-Logistic Regression ##########################
    # dataset_path2 = r"C:\Users\1000303969\OneDrive - Western Digital\work\tdp classification\data\increase_decrease\high_low_other"
    # clip_trainer = CLIP_Trainer(dataset_path)
    # clip_trainer.train_model(model_method="logistic")