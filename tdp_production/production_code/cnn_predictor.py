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

## seed
torch.manual_seed(42)
np.random.seed(42)

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
        pred = output.argmax(-1).item()
        class_to_idx = self.chpt["class_to_idx"]
        idx_to_class = {v:k for k,v in class_to_idx.items()}
        confi = output.softmax(-1).detach().cpu().numpy().tolist()[0][pred]
        percent_confi = round(confi*100, 2)
        result = {
            "class": idx_to_class[pred],
            "confidence": percent_confi
        }
        return result
    
class FIG_TO_ARR:
    def __init__(self, fig):
        assert isinstance(fig, plt.Figure), "fig must be a matplotlib figure"
        self.fig = fig

    def fig2rgb_array(self):
        self.fig.canvas.draw()
        buf = self.fig.canvas.tostring_rgb()
        ncols, nrows = self.fig.canvas.get_width_height()
        return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    
    def fig_to_arr(self):
        img_arr = self.fig2rgb_array()
        return img_arr