from cnn_trainer import CNN_Trainer
from cnn_predictor import CNN_Predictor
from clip_trainer import CLIP_Trainer

if __name__ == "__main__":
    dataset_path = ""
    
    trainer = CNN_Trainer(dataset_path)

    ########################## NORMAL TRAINING ##########################
    params_v2 = {
        "model_name": "resnet26d",
        "EPOCHS": 25,
        "use_lookahead": True,
        "SAVED": False,
        "num_accumulate":4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "model_version": "recovery_v1",
        "batch_size": 32,
        "valid_size":0.4,
        "test_size":0.4,
        "sched":"cosine",
        "opt":"adamw",
        "use_wandb":False,
    }
    trainer.train_model_v2(**params_v2)

    ########################## CROSS VALIDATION ##########################
    # params_validate = {
    #     "model_name": "resnet50d",
    #     "num_epochs": 20,
    #     "train_batch_size": 32,
    #     "eval_batch_size": 64,
    #     "k_splits": 10,
    #     "num_accumulate": 6,
    # }
    # trainer.train_cross_validation(**params_validate)

    ########################## PREDICTION ##########################
    # model_path = ""
    # image_path = ""
    # # image_path = ""
    # folder_path = ""
    
    # # # image_path = ""
    
    # predictor = CNN_Predictor(model_path)

    # model = predictor.load_model()
    # result = predictor.predict(model, image_path)
    # print(f"Class prediction: {result['class']} | Confidence: {result['confidence']}")

    # predictor.predict_folder(model, folder_path)    

    ########################## CLIP-Logistic Regression ##########################
    # dataset_path2 = ""
    # clip_trainer = CLIP_Trainer(dataset_path)
    # clip_trainer.train_model(model_method="logistic")
