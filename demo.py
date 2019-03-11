import sys
import cv2
import torch
import numpy as np

from models import Age, Gender



def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    img = img[None, :, :, :]
    tensor = torch.from_numpy(img)
    tensor = tensor.type('torch.FloatTensor')
    return tensor


def expected_age(vector):
    res = [(i+1)*v for i, v in enumerate(vector)]
    return sum(res)


# load and setup model
gender_model = torch.load('pth/gender.pth')
gender_model.eval()
age_model = torch.load('pth/age.pth')
age_model.eval()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python demo.py path/to/img")
        sys.exit()
    
    path = sys.argv[1]
    tensor = preprocess(path)

    with torch.no_grad():
        gender = gender_model(tensor)
        gender = gender.numpy().squeeze()

        age = age_model(tensor)
        age = age.numpy().squeeze()
        age = expected_age(age)

        print("predict image: {}".format(path))
        print("woman: {:.3f}, man: {:.3f}".format(gender[0], gender[1]))
        print("age: {:.3f}".format(age))
