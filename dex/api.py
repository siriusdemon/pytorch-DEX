import os
import cv2
import torch
import numpy as np

from .models import Age, Gender

age_model = Age()
gender_model = Gender()

cwd = os.path.dirname(__file__)
age_model_path = os.path.join(cwd, 'pth/age_sd.pth')
gender_model_path = os.path.join(cwd, 'pth/gender_sd.pth')

def _eval():
    global age_model
    global gender_model
    age_model.load_state_dict(torch.load(age_model_path))
    age_model.eval()
    gender_model.load_state_dict(torch.load(gender_model_path))
    gender_model.eval()
    

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    img = img[None, :, :, :]
    tensor = torch.from_numpy(img)
    tensor = tensor.type('torch.FloatTensor')
    return tensor


def expected_age(vector):
    res = [(i+1)*v for i, v in enumerate(vector)]
    return sum(res)


def estimate_age(img):
    if type(img) == str:
        img = cv2.imread(img)
    tensor = preprocess(img)
    with torch.no_grad():
        output = age_model(tensor)
    output = output.numpy().squeeze()
    age = expected_age(output)
    return age

def estimate_gender(img):
    if type(img) == str:
        img = cv2.imread(img)
    tensor = preprocess(img)
    with torch.no_grad():
        output = gender_model(tensor)
    output = output.numpy().squeeze()
    return output[0], output[1]

def estimate(img):
    """return values as (age, female, male)"""
    img = cv2.imread(img)
    result = [estimate_age(img)]
    result.extend(estimate_gender(img))
    return result
    