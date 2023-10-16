from importTDP_lib import *

class CLIP_Trainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def load_dataset(self):
        dataset = ImageFolder(self.dataset_path, transform=self.preprocess)
        return dataset

    def prep_dataloader(self, dataset, batch_size=64 ,valid_size=0.5):
        train_indices, test_indices = train_test_split(range(len(dataset)), test_size=valid_size, stratify=dataset.targets, random_state=42)

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        batch_size = batch_size 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        ## print logging
        print("===== Dataset Summary =====")
        print("Classes: ", dataset.classes)
        print(f"Total Data:  {len(dataset)}")
        print(f"Train Data:  {len(train_dataset)}")
        print(f"Test Data: {len(test_dataset)} \n")

        return train_loader, test_loader
    
    def logistic_model(self, train_features, train_labels):
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, solver="saga")
        print("Training...")
        classifier.fit(train_features, train_labels)
        return classifier

    def xgboost_model(self, train_features, train_labels):
        classifier = XGBClassifier()
        print("Training...")
        classifier.fit(train_features, train_labels)
        return classifier

    def get_features(self, dataloader):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                features = self.model.encode_image(images.to(self.device))
                all_features.append(features)
                all_labels.append(labels)
            
        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
    
    def train_model(self, model_method):
        dataset = self.load_dataset()

        train_loader, test_loader = self.prep_dataloader(dataset)
        print("===== Extract Feature =====\n")
        train_features, train_labels = self.get_features(train_loader)
        test_features, test_labels = self.get_features(test_loader)

        if model_method == "logistic":
            classifier = self.logistic_model(train_features, train_labels)

        print("===== Evaluation =====")
        predictions = classifier.predict(test_features)
        accuracy = np.mean((test_labels == predictions).astype("float32")) * 100.
        print(f"Accuracy: {accuracy}")
        print(classification_report(test_labels, predictions, target_names=dataset.classes))
        ## print missclassified
        i = 0
        print('===== MISCLASSIFIED =====')
        for t, p in zip(test_labels, predictions):
            misname = os.path.basename(dataset.samples[i][0])
            print(f"+++++++File {misname} | Target: {dataset.classes[t]} | Predicted: {dataset.classes[p]}")
            if t != p:
                print(t, p)
                print(f"======File {misname} | Target: {dataset.classes[t]} | Predicted: {dataset.classes[p]}")
            i += 1