# Imports

import torch
import torchvision
from torch import mm
import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import*
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50,deeplabv3_resnet101
from torchvision.models import (vgg16,vgg16_bn,densenet121,resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d)

import io
import os
import cv2
import json
import math
import copy
import time
import kornia
import shutil
import pickle
import mlflow
import random
import skimage
import logging
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import sqrt
import mlflow.pytorch
# import face_recognition
from scipy import stats
from pathlib import Path
from ast import literal_eval
import albumentations as albu
from datetime import datetime
import matplotlib.pyplot as plt
from pprint import PrettyPrinter
from torchsummary import summary
from os.path import isfile, join
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import ImageDraw, ImageFont
from itertools import product as product
from albumentations import pytorch as AT
# import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,f1_score
