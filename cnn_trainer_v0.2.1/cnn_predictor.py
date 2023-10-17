from importTDP_lib import *

class CNN_Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.device == torch.device("cpu"):
            self.chpt = torch.load(self.model_path, map_location=torch.device('cpu'))
        else:
            self.chpt = torch.load(self.model_path)
    
    def load_model(self):
        model = timm.create_model(self.chpt["arch"], pretrained=True, num_classes=len(self.chpt["class_to_idx"])).to(self.device)
        if self.device == "cpu":
            model.load_state_dict(self.chpt["state_dict"], map_location=torch.device('cpu'))
        else:
            model.load_state_dict(self.chpt["state_dict"])

        model = model.eval()
        return model
    
    def predict_folder(self, model, folder_path)-> None:
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            result_json = self.predict(model, image_path)
            print(f"File {os.path.basename(image_path)}, Class prediction: {result_json['class']} | Confidence: {result_json['confidence']}")
    
    def predict(self, model, image_path):
        img = Image.open(image_path).convert("RGB")
        transform = self.chpt["transform"]
        img = transform(img).unsqueeze(0).to(self.device)
        output = model(img)
        pred = output.argmax(-1).item()
        class_to_idx = self.chpt["class_to_idx"]
        idx_to_class = {v:k for k,v in class_to_idx.items()}
        result = {
            "class": idx_to_class[pred],
            "confidence": output.softmax(-1).detach().cpu().numpy().tolist()[0][pred]
        }
        return result