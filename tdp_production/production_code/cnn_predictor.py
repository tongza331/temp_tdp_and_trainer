import os
import cv2 
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import timm
import seaborn as sns
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

## seed
torch.manual_seed(42)
np.random.seed(42)

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
        model.load_state_dict(self.chpt["state_dict"])
        model = model.eval()
        return model
    
    def convert_to_arr(self, fig):
        assert isinstance(fig, plt.Figure), "fig must be a matplotlib figure"
        fig_to_arr = FIG_TO_ARR(fig)
        image_arr = fig_to_arr.fig_to_arr()
        return image_arr

    def predict(self, model, image_path, is_path=False):
        if is_path:
            img = Image.open(image_path).convert("RGB")
        else:
            img = self.convert_to_arr(image_path)
            img = Image.fromarray(img).convert("RGB")
        transform = self.chpt["transform"]
        img = transform(img).unsqueeze(0).to(self.device)
        
        output = model(img)
        ## predict top 3
        pred = output.topk(3, dim=-1).indices.squeeze().tolist()
        class_to_idx = self.chpt["class_to_idx"]
        idx_to_class = {v:k for k,v in class_to_idx.items()}
        ## confidence of each k
        confi = output.softmax(-1).detach().cpu().numpy().tolist()[0]
        percent_confi = [round(i*100, 2) for i in confi]
        result = {
            "class": [idx_to_class[i] for i in pred],
            "confidence": [percent_confi[i] for i in pred]
        }
        return result

    def _one_ensemble_pred(self, model_path, fig):
        pass
    
    def predict_weight_ensemble_mpl(self, model_path_list, image_path, is_path=False):
        n_models = len(model_path_list)
        ## apply multiprocess
        

    def predict_weight_ensemble(self, model_path_list, image_path, is_path=False):
        ## load model from model_dict
        n_models = len(model_path_list)
        for idx_model in range(n_models):
            if is_path:
                img = Image.open(image_path).convert("RGB")
            else:
                img = self.convert_to_arr(image_path)
                img = Image.fromarray(img).convert("RGB")
                
            print(f"Loading model {idx_model+1}/{n_models}")
            
            if self.device == torch.device("cpu"):
                chpt = torch.load(model_path_list[idx_model], map_location=torch.device('cpu'))
            else:
                chpt = torch.load(model_path_list[idx_model])
                
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