import os
import cv2 
import numpy as np
import pandas as pd
import random

## evaluation/split data libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

## model libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier

# pretrained models
import clip
import timm
import timm.optim
import timm.scheduler

## plot libraries
import seaborn as sns
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

## other libraries
import time
import glob
import wandb
import warnings
warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm as tqdm_nb

## custom libraries
from earlystopping import EarlyStopping
# from _extractTDP_features import ExtractTDP

from sklearn.model_selection import KFold, StratifiedKFold
import evaluate
import timm.optim
import timm.scheduler
from timm.data import ImageDataset, create_dataset, create_loader
from timm.data.transforms_factory import create_transform

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MulticlassROC
import threading

## seed
torch.manual_seed(42)
np.random.seed(42)