from importTDP_lib import *
from natsort import natsorted

class CNN_Predictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if model_path is not None:
            self.model_path = model_path
            
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
    
    def convert_to_arr(self, fig):
        assert isinstance(fig, plt.Figure), "fig must be a matplotlib figure"
        fig_to_arr = FIG_TO_ARR(fig)
        image_arr = fig_to_arr.fig_to_arr()
        return image_arr
    
    def predict_folder(self, model, folder_path)-> None:
        for file in natsorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, file)
            result_json = self.predict(model, image_path)
            print(f"File {os.path.basename(image_path)}, Class prediction: {result_json['class']} | Confidence: {result_json['confidence']}")
    
    def predict(self, model, image_path):
        img = Image.open(image_path).convert("RGB")
        transform = self.chpt["transform"]
        img = transform(img).unsqueeze(0).to(self.device)
        output = model(img)
        # pred = output.argmax(-1).item()
        ## pred top-3
        preds = output.topk(3, dim=-1).indices.squeeze(0).tolist()
        class_to_idx = self.chpt["class_to_idx"]
        idx_to_class = {v:k for k,v in class_to_idx.items()}
        result = {
            "class": [idx_to_class[pred] for pred in preds],
            "confidence": output.softmax(-1).detach().cpu().numpy().tolist()[0][preds[0]]
        }
        return result
    
    def predict_weight_ensemble(self, model_dict, image_path, is_path=True):
        ## load model from model_dict
        n_models = len(model_dict)
        
        for idx_model in range(n_models):
            if is_path:
                img = Image.open(image_path).convert("RGB")
            else:
                img = Image.fromarray(image_path).convert("RGB")
                
            print(f"Loading model {idx_model+1}/{n_models}")
            if self.device == torch.device("cpu"):
                chpt = torch.load(model_dict[idx_model], map_location=torch.device('cpu'))
            else:
                chpt = torch.load(model_dict[idx_model])
            model = timm.create_model(chpt["arch"], pretrained=True, num_classes=len(chpt["class_to_idx"])).to(self.device)
            model.load_state_dict(chpt["state_dict"])
            
            transform = chpt["transform"]
            img = transform(img).unsqueeze(0).to(self.device)
            output = model(img)
            if idx_model == 0:
                output_sum = output
            else:
                output_sum += output
            
        ## map idx to class
        preds = output_sum.topk(3, dim=-1).indices.squeeze(0).tolist()
        class_to_idx = chpt["class_to_idx"]
        idx_to_class = {v:k for k,v in class_to_idx.items()}
        result = {
            "class": [idx_to_class[pred] for pred in preds],
            "confidence": output_sum.softmax(-1).detach().cpu().numpy().tolist()[0][preds[0]]
        }
        return result
    
class FIG_TO_ARR:
    def __init__(self, fig):
        assert isinstance(fig, plt.Figure), "fig must be a matplotlib figure"
        self.fig = fig

    def fig2rgb_array(self):
        self.fig.tight_layout(pad=0)
        self.fig.canvas.draw()
        buf = self.fig.canvas.tostring_rgb()
        ncols, nrows = self.fig.canvas.get_width_height()
        return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    
    def fig_to_arr(self):
        img_arr = self.fig2rgb_array()
        return img_arr