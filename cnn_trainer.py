from importTDP_lib import *
import threading

class CNN_Trainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.transform = self.create_transform(CUSTOM=False, input_size=(224,224))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## initialize list
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def load_dataset(self):
        dataset = ImageFolder(self.dataset_path, transform=self.transform)
        return dataset

    def all_loader(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        return loader
    
    def _normalize_valie(dataloader):
        batch, sum_, sqr_ = 0, 0, 0
        for x, y in dataloader:
            sum_ += torch.mean(x, axis=[0,2,3])
            sqr_ += torch.mean(x**2, axis=[0,2,3])
            batch += 1
        mean = sum_ / batch
        std = (sqr_/batch)-mean**2
        print("mean, std", mean, std)
        return mean, std

    def custome_normalize(self):
        dl = self.all_loader(self.load_dataset())
        mean, std = self._normalize_valie(dl)
        return mean, std
    
    def get_filename(self, dataset, test_indices):
        filename_list = []
        for i in test_indices:
            filename_list.append(dataset.samples[i][0])
        return filename_list

    def prep_dataloader(self, dataset, batch_size=32 ,valid_size=0.5, test_size=0.5):
        labels = [label for _, label in dataset.samples]

        train_indices, rest_indices = train_test_split(range(len(dataset)), test_size=valid_size, stratify=labels, random_state=42)
        val_indices, test_indices = train_test_split(rest_indices, test_size=test_size, stratify=[labels[i] for i in rest_indices], random_state=42)

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        batch_size = batch_size 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        ## print logging
        print("===== Dataset Summary =====")
        print("Classes: ", dataset.classes)
        print(f"Total Data:  {len(dataset)}")
        print(f"Train Data:  {len(train_dataset)}")
        print(f"Validation Data: {len(val_dataset)}")
        print(f"Test Data: {len(test_dataset)} \n")

        return train_loader, val_loader, test_loader

    def create_transform(self, CUSTOM=False, input_size=(224,224)):
        if CUSTOM:
            mean, std = self.custome_normalize()
            transform = Compose([Resize(input_size), ToTensor(), Normalize(mean, std)])
        else:
            transform = Compose([Resize(input_size, interpolation=Image.LANCZOS), 
                                 ToTensor(), 
                                #  Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                                 Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
        
        return transform

    def create_model(self, model_name, num_classes, pretrained=True):
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        model.to(self.device)
        model.eval()
        return model

    def train_model_v2(self, 
                model_name, EPOCHS, 
                use_lookahead=False, 
                SAVED=False, 
                num_accumulate=4, 
                lr=1e-4, 
                weight_decay=1e-3, 
                model_version="", 
                batch_size=32,
                valid_size=0.5,
                test_size=0.5,
                sched="cosine",
                opt="adamw",
                use_wandb=False,
                patience=7
    ):
        ## intialize wandb
        if use_wandb:
            wandb.init(project="tdp_classification", name=f"{model_name}_{model_version}")

        early_stopping = EarlyStopping(patience=patience, verbose=True, path=f"models/best_{model_name}_{model_version}_checkpoint.pt", SAVED=SAVED)
                
        dataset = self.load_dataset()
        train_loader, val_loader, test_loader = self.prep_dataloader(dataset, batch_size=batch_size, valid_size=valid_size, test_size=test_size)

        num_accumulate = num_accumulate
        num_classes = len(dataset.classes)

        model = self.create_model(model_name, num_classes=num_classes, pretrained=True)

        metric = evaluate.load("accuracy")
        optimizer = timm.optim.create_optimizer_v2(model, opt=opt, lr=lr, weight_decay=weight_decay)

        if use_lookahead:
            optimizer = timm.optim.Lookahead(optimizer, alpha=0.5, k=6)
        
        savedir = "models"
        os.makedirs(savedir, exist_ok=True)
        
        criterion = nn.CrossEntropyLoss()
        scheduler = timm.scheduler.create_scheduler_v2(sched=sched ,optimizer=optimizer, num_epochs=EPOCHS, min_lr=1e-6, plateau_mode="min", patience_epochs=5)[0]

        all_eval_scores = []
        info = {
            "metric_train": [],
            "metric_val": [],
            "train_loss": [],
            "val_loss": [],
            "best_metric_val": -999,
            "min_metric_val": 999,
        }

        if use_wandb:
            wandb.watch(model, criterion, log="all", log_freq=5)

        print("===== Training =====")
        for epoch in range(EPOCHS):
            train_loss_epoch = []
            val_loss_epoch = []

            train_preds = []
            train_targets = []

            val_preds = []
            val_targets = []

            test_preds = []
            test_targets = []

            num_updates = epoch * len(train_loader)
            print(f"Epoch: {epoch+1}/{EPOCHS}")

            ## Training loop
            model.train()
            for idx, batch in enumerate(tqdm(train_loader)):
                inputs, targets = batch
                outputs = model(inputs.to(self.device))
                loss = criterion(outputs, targets.to(self.device))

                loss.backward()

                if ((idx+1) % num_accumulate==0) or (idx + 1 == len(train_loader)):
                    optimizer.step()
                    scheduler.step_update(num_updates=num_updates)
                    optimizer.zero_grad()

                train_loss_epoch.append(loss.item())
                train_preds += outputs.argmax(-1).detach().cpu().numpy().tolist()
                train_targets += targets.tolist()

            if use_lookahead:
                optimizer.sync_lookahead()
            
            if sched == "plateau":
                scheduler.step(epoch,loss)
            elif sched == "cosine":
                scheduler.step(epoch=epoch + 1)

            ## Eval loop
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    inputs, targets = batch
                    outputs = model(inputs.to(self.device))
                    loss = criterion(outputs, targets.to(self.device))

                    # Log Values
                    val_loss_epoch.append(loss.item())
                    val_preds += outputs.argmax(-1).detach().cpu().numpy().tolist()
                    val_targets += targets.tolist()

                metric_train = metric.compute(predictions=train_preds, references=train_targets)["accuracy"]
                metric_val = metric.compute(predictions=val_preds, references=val_targets)["accuracy"]

                info["metric_train"].append(metric_train)
                info["metric_val"].append(metric_val)

                info["train_loss"].append(np.average(train_loss_epoch))
                info["val_loss"].append(np.average(val_loss_epoch))

                if metric_val > info["best_metric_val"]:
                    print("New Best Score!")
                    info["best_metric_val"] = metric_val

                    model_output_path = os.path.join(savedir, "best_{}_{}.pt".format(model_name, model_version))
                    
                    if SAVED:
                        print("Saving model...")
                        torch.save({
                            "arch":model_name,
                            "state_dict": model.state_dict(),
                            "class_to_idx": dataset.class_to_idx,
                            "transform": self.transform
                        }, model_output_path)

            if use_wandb:
                wandb.log({"train_loss": np.average(train_loss_epoch), "val_loss": np.average(val_loss_epoch), "train_acc": metric_train, "val_acc": metric_val})
            
            print(f"Epoch: {epoch+1}/{EPOCHS} | Train Accuracy: {metric_train} | Train Loss: {np.average(train_loss_epoch)} | Validation Accuracy: {metric_val} | Validation Loss: {np.average(val_loss_epoch)}")
            
            early_stopping(info["val_loss"][-1], model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        all_eval_scores.append(info["best_metric_val"])
        self.plot_learning_curve(model_name, info, model_version)
        print("Fininshed Training.")

        print("===== Evaluation =====")
        if SAVED:
            chpt = torch.load(model_output_path)
            loaded_model = timm.create_model(chpt["arch"], pretrained=True, num_classes=len(chpt["class_to_idx"])).to(self.device)
            loaded_model.load_state_dict(chpt["state_dict"])
            loaded_model = loaded_model.to(self.device)
        else:
            loaded_model = model

        loaded_model.eval()
        i = 0
        print('===== MISCLASSIFIED =====')
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                outputs = loaded_model(inputs.to(self.device))
                targets = targets.to(self.device)

                # Log Values
                test_preds += outputs.argmax(-1).detach().cpu().tolist()
                test_targets += targets.detach().cpu().tolist()

                _, preds = torch.max(outputs, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    if t != p:
                        print(f"Image {i} | Target: {dataset.classes[t]} | Predicted: {dataset.classes[p]}")
                    i += 1

        print("=========== CONFUSION MATRIX ===========")
        cm = confusion_matrix(test_targets, test_preds)
        df_cm = self.cm2df(cm, dataset.classes)
        print(df_cm)

        print("=========== CLASSIFICATION REPORT ===========")
        print(classification_report(test_targets, test_preds, target_names=dataset.classes))

        #### OPTIONAL: SAVE LAST MODEL ####
        choice_input = input("Do you want to save the last model? (y/n): ")
        if choice_input == "y":
            model_output_path = os.path.join(savedir, f"last_{model_name}_{model_version}.pt")
            torch.save({
                "arch":model_name,
                "state_dict": model.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "transform": self.transform
            }, model_output_path)
        else:
            pass

    def train_cross_validation(self, model_name, num_epochs, train_batch_size=8, eval_batch_size=12, k_splits=5, num_accumulate=4, model_version="cross_validate"):
        os.makedirs("models", exist_ok=True)
        all_eval_scores = []

        kf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)
        # kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
        criterion = nn.CrossEntropyLoss()
        metric = evaluate.load("f1")

        dataset = self.load_dataset()

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, dataset.targets)):
            print(f"Fold: {fold+1} of {k_splits}")

            # Load Model
            model = timm.create_model(model_name, pretrained=True, num_classes=len(dataset.classes)).to(self.device)

            # Load Optimizer and Scheduler
            optimizer = timm.optim.create_optimizer_v2(model, opt="adamw", lr=1e-3)
            optimizer = timm.optim.Lookahead(optimizer, alpha=0.5, k=6)

            scheduler = timm.scheduler.create_scheduler_v2(optimizer, num_epochs=num_epochs)[0]

            # Load Data
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)

            train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

            # Reset Model Info
            info = {
                "metric_train": [],
                "metric_val": [],
                "train_loss": [],
                "val_loss": [],
                "best_metric_val": -999,
                "min_metric_val": 999,
            }

            for epoch in range(num_epochs):
                train_loss_epoch = []
                val_loss_epoch = []

                train_preds = []
                train_targets = []

                val_preds = []
                val_targets = []

                num_updates = epoch * len(train_dataloader)

                ## Training loop
                model.train()
                for idx, batch in enumerate(tqdm(train_dataloader)):
                    inputs, targets = batch
                    outputs = model(inputs.to(self.device))
                    loss = criterion(outputs, targets.to(self.device))

                    loss.backward()

                    if ((idx+1) % num_accumulate==0) or (idx + 1 == len(train_dataloader)):
                        optimizer.step()
                        scheduler.step_update(num_updates=num_updates)
                        optimizer.zero_grad()
                    
                    train_loss_epoch.append(loss.item())
                    train_preds += outputs.argmax(-1).detach().cpu().tolist()
                    train_targets += targets.tolist()
                
                optimizer.sync_lookahead()
                scheduler.step(epoch + 1)

                ## Eval loop
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(val_dataloader):
                        inputs, targets = batch
                        outputs = model(inputs.to(self.device))
                        loss = criterion(outputs, targets.to(self.device))
                        # Log Values
                        val_loss_epoch.append(loss.item())
                        val_preds += outputs.argmax(-1).detach().cpu().tolist()
                        val_targets += targets.tolist()
                
                # Log Data
                metric_train = metric.compute(predictions=train_preds, references=train_targets, average="macro")["f1"]
                metric_val = metric.compute(predictions=val_preds, references=val_targets, average="macro")["f1"]

                info["metric_train"].append(metric_train)
                info["metric_val"].append(metric_val)

                info["train_loss"].append(np.average(train_loss_epoch))
                info["val_loss"].append(np.average(val_loss_epoch))

                if metric_val > info["best_metric_val"] or val_loss_epoch[-1] < info["min_metric_val"]:
                    print("New Best Score!")
                    info["best_metric_val"] = metric_val
                    info["min_metric_val"] = val_loss_epoch[-1]
                    
                    model_output_path = os.path.join("models", f"best_{model_name}_{fold+1}.pt")
                    torch.save({
                        "arch":model_name,
                        "state_dict": model.state_dict(),
                        "class_to_idx": dataset.class_to_idx,
                        "transform": self.transform
                    }, model_output_path)

                # print(info)
                print(f"Fold: {fold+1} | Epoch: {epoch+1} | Metric (f1): {metric_val} | Train Loss: {np.average(train_loss_epoch)} | Validation Loss: {np.average(val_loss_epoch)}")

            all_eval_scores.append(info["best_metric_val"])
            self.plot_learning_curve(model_name, info, model_version, mode="cross_validate", idx=fold+1)

            ## plot graph of each k-fold
            fig, ax = plt.subplots(figsize=(15,5))
            ax.plot(all_eval_scores, marker="o", color="red")
            ax.set_title("Validation Score")
            ax.set_xlabel("Fold")
            ax.set_ylabel("F1 Score")
            ax.set_xticks(range(k_splits))
            ax.set_xticklabels(range(1, k_splits+1))
            plt.savefig(f"K-FOLD validation_score_{model_name}.jpg")

        print("======= CROSS VALIDATION EVALUATION =======")
        for fold in range(k_splits):
            chpt = torch.load(os.path.join("models", f"best_{model_name}_{fold+1}.pt"))
            loaded_model = timm.create_model(chpt["arch"], pretrained=True, num_classes=len(chpt["class_to_idx"])).to(self.device)
            loaded_model.load_state_dict(chpt["state_dict"])
            loaded_model.eval()
            predictions = []
            references = []

            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, targets = batch
                    outputs = loaded_model(inputs.to(self.device))
                    targets = targets.to(self.device)

                    # Log Values
                    predictions += outputs.argmax(-1).detach().cpu().tolist()
                    references += targets.detach().cpu().tolist()
            
            print(f"=========== Fold: {fold+1} ===========")
            cm = confusion_matrix(references, predictions)
            df_cm = self.cm2df(cm, dataset.classes)
            print(df_cm)
            print(classification_report(references, predictions, target_names=dataset.classes))    

    def cm2df(self, cm, labels):
        data = []

        for i, row_label in enumerate(labels):
            row_data = {}
            for j, col_label in enumerate(labels):
                row_data[col_label] = cm[i, j]
            data.append(row_data)
        df = pd.DataFrame(data, index=labels)
        return df
    
    def plot_learning_curve(self, model_name, info, model_version, mode="", idx=0):
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        ax[0].plot(info["train_loss"], label="train_loss", marker="o", color="red")
        ax[0].plot(info["val_loss"], label="val_loss", marker="o", color="blue")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(info["metric_train"], label="train_acc", marker="o", color="red")
        ax[1].plot(info["metric_val"], label="val_acc", marker="o", color="blue")
        ax[1].set_title("Accuracy")
        ax[1].legend()

        if mode == "cross_validate":
            plt.savefig(f"loss_acc_graph_{model_name}_{model_version}_fold{idx}.jpg")
        else:
            plt.savefig(f"loss_acc_graph_{model_name}_{model_version}.jpg")
        
        plt.close("all")
