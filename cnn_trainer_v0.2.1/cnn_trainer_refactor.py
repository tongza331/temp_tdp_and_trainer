from importTDP_lib import *

class CNN_Trainer:
    
    def __init__(self, dataset_path, 
                 transform_list:list=[Resize((224, 224), interpolation=Image.LANCZOS), ToTensor()],
                 dataset_type="folder",
                 model_save_path="models",
                 use_mix_precision=False):
        
        self.dataset_path = dataset_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = Compose(transform_list)
        self.dataset_type = dataset_type
        self.model_save_path = model_save_path
        self.use_mix_precision = use_mix_precision
        
        os.makedirs(self.model_save_path, exist_ok=True)

        self.train_acc_list, self.train_loss_list = [], []
        self.valid_acc_list, self.valid_loss_list = [], []
        
    def load_dataset(self):
        try:
            if self.dataset_type == "folder":
                dataset = ImageFolder(root=self.dataset_path, transform=self.transform)
        except ValueError as error:
            return error, "Dataset type not supported. -> Please use 'folder' type."
        
        return dataset
    
    def prep_dataloader(self, dataset, batch_size=32, valid_size=0.3, test_size=0.2):
        if self.dataset_type == "folder":
            labels = [label for _, label in dataset.samples]

            train_indices, rest_indices = train_test_split(range(len(dataset)), test_size=valid_size, stratify=labels, random_state=42)
            val_indices, test_indices = train_test_split(rest_indices, test_size=test_size, stratify=[labels[i] for i in rest_indices], random_state=42)

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
        return train_loader, val_loader, test_loader
    
    def create_model(self, model_name, num_classes, pretrained=True):
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        model = model.to(self.device)
        model.eval()
        return model
    
    def training_loop(self):
        self.model.train()
        torch.set_grad_enabled(True)
        
        for idx, batch in enumerate(tqdm(self.train_loader)):
            inputs, labels = batch
            outputs = self.model(inputs.to(self.device))
            loss = self.criteria(outputs, labels.to(self.device))
            loss.backward()
            
            if self.use_lookahead:
                if ((idx+1) % self.n_accumulate==0) or (idx + 1 == len(self.train_loader)):
                    self.optimizer.step()
                    self.scheduler.step_update(self.num_updates)
                    self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.use_lookahead:
                self.optimizer.sync_lookahead()
                
            self.train_loss_list.append(loss.item())
            self.train_preds += outputs.argmax(-1).detach().cpu().numpy().tolist()
            self.train_targets += labels.tolist()
            
    
    def validation_loop(self):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                inputs, labels = batch
                outputs = self.model(inputs.to(self.device))
                loss = self.criteria(outputs, labels.to(self.device))
                
                self.val_loss_epoch.append(loss.item())
                self.val_preds += outputs.argmax(-1).detach().cpu().numpy().tolist()
                self.val_targets += labels.tolist()
                
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        elif isinstance(self.scheduler, timm.scheduler.CosineLRScheduler):
            self.scheduler.step()
        
    
    def __parameter_init(self):
        if self.patience is not None:
            self.early_stopping = EarlyStopping(patience=self.patience, verbose=True, SAVED=self.SAVED_BEST)
        
        # Initialize dataset
        dataset = self.load_dataset()
        self.train_loader, self.val_loader, self.test_loader = self.prep_dataloader(dataset, batch_size=self.batch_size, valid_size=self.valid_size, test_size=self.test_size)
        
        # Initialize model
        self.model = self.create_model(self.model_name, num_classes=len(dataset.classes), pretrained=True)
        
        # Initialize metric, optimizer and scheduler
        self.metric = evaluate.load("accuracy")
        self.optimizer = timm.optim.create_optimizer_v2(self.model, lr=self.lr, weight_decay=self.weight_decay)
        # self.scheduler = timm.scheduler.create_scheduler_v2(sched=self.scheduler, optimizer=self.optimizer, num_epochs=self.EPOCHS)[0]
        try:
            if self.scheduler_name == "cosine":
                self.scheduler = timm.scheduler.CosineLRScheduler(
                                                                    self.optimizer,
                                                                    t_initial=self.EPOCHS,
                                                                    cycle_decay=0.1,
                                                                    lr_min=1e-6,
                                                                    warmup_t=3,
                                                                    warmup_lr_init=self.lr,
                                                                    cycle_limit=1,
                                                                )
            elif self.scheduler_name == "reduce_plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", patience=5, verbose=True)
            else:
                print("LR Scheduler is not define.")
        except ValueError as error:
            return error, "Scheduler name not supported. -> Please use 'cosine' or 'reduce_plateau'."
        
        self.criteria = nn.CrossEntropyLoss()
    
    def train_model(self,
                    model_name,
                    EPOCHS,
                    lr=1e-4,
                    weight_decay=1e-3,
                    model_version="",
                    batch_size=32,
                    valid_size=0.3,
                    test_size=0.2,
                    scheduler_name=None,
                    optimizer=None,
                    patience=None,
                    SAVED_BEST=False,
                    use_lookahead=False,
                    n_accumulate=5,
                    alpha=0.5,
                    k=6
                    ):
        # Initialize early stopping
        self.model_name = model_name
        self.EPOCHS = EPOCHS
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_version = model_version
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.scheduler_name = scheduler_name
        self.optimizer = optimizer
        self.patience = patience
        self.SAVED_BEST = SAVED_BEST
        self.use_lookahead = use_lookahead
        self.n_accumulate = n_accumulate
        self.alpha = alpha
        self.k = k
        
        self.version_folder = os.path.join(self.model_save_path, self.model_version)
        os.makedirs(self.version_folder, exist_ok=True)
        
        
        self.__parameter_init()
        
        all_eval_scores = []
        info = {
            "metric_train": [],
            "metric_val": [],
            "loss_train": [],
            "loss_val": [],
            "best_metric_val": -999,
            "min_metric_val": 999
        }
        
        for epoch in range(self.EPOCHS):
            self.train_loss_epoch = []
            self.val_loss_epoch = []

            self.train_preds = []
            self.train_targets = []

            self.val_preds = []
            self.val_targets = []

            self.test_preds = []
            self.test_targets = []
            
            self.num_updates = epoch * len(self.train_loader)
            
            self.training_loop()
            self.validation_loop()
            
            metric_train = self.metric.compute(predictions=self.train_preds, references=self.train_targets)["accuracy"]
            metric_val = self.metric.compute(predictions=self.val_preds, references=self.val_targets)["accuracy"]
            
            info["metric_train"].append(metric_train)
            info["metric_val"].append(metric_val)
            info["loss_train"].append(np.average(self.train_loss_epoch))
            info["loss_val"].append(np.average(self.val_loss_epoch))
            
            self.early_stopping(metric_val, self.model)
            early_save_flag = self.early_stopping.SAVE_FLAG
            
            save_info = {
                "metric_val": metric_val,
                "best_metric_val": info["best_metric_val"],
                "early_counter": self.early_stopping.counter,
                "early_save_flag": early_save_flag,
                "patience": self.patience,
                "training_info": {
                    "metric_train": metric_train,
                    "metric_val": metric_val,
                    "loss_train": np.average(self.train_loss_epoch),
                    "loss_val": np.average(self.val_loss_epoch)
                }
            }
            
            self.__save_model(self.model, **save_info)
            
            print(f"Epoch: {epoch+1}/{self.EPOCHS} | Train loss: {np.average(self.train_loss_epoch):.4f} | Train acc: {metric_train:.4f} | Val loss: {np.average(self.val_loss_epoch):.4f} | Val acc: {metric_val:.4f}")
            
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
    
    def train_cross_validation(self):
        pass
    
    def __save_model(self, model, **save_info):
        if (save_info["metric_val"] > save_info["best_metric_val"]) or save_info["early_save_flag"] and save_info["early_counter"] <= save_info["patience"]:
            print("New best model saved.")
            
            model_output_path = os.path.join(self.model_save_path, f"best_{self.model_name}_{self.model_version}.pt")
            
            if self.SAVED_BEST:
                torch.save({
                    "arch": self.model_name,
                    "state_dict": model.state_dict(),
                    "class_to_idx": self.train_loader.dataset.class_to_idx,
                    "transform": self.transform,
                    "training_info": save_info["training_info"]
                }, model_output_path)
    
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
    
    def _cm2df(self):
        pass
    
    
    
    