import torch
import torch.nn as nn
import torchvision
import json

from attacks import * 
from utils import *
from models import *

input_dir = './Images'
output_dir = './Examples'
imgfiles = ['list of image file names. e.g.', 'myimage.png']
imglabels = ['list of labels as int']
imgtargets = ['list of targets as int']

if __name__ == "__main__":
    os.makedirs("./Examples", exist_ok=True)
    input_dir += "/" * (input_dir[-1] != "/")
    output_dir += "/" * (output_dir[-1] != "/")

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    modelResNet = torchvision.models.resnet50(weights='DEFAULT')
    modelVGG = torchvision.models.vgg19(weights='DEFAULT')
    modelResNet.eval().to(device)
    modelVGG.eval().to(device)

    CAMResNet1 = nn.Sequential(*list(modelResNet.children())[:-2])
    CAMResNet2 = CAMNet(latent_dim=2048).to(device)
    CAMResNet = nn.Sequential(CAMResNet1, CAMResNet2)
    if use_gpu:
        state_dict = torch.load("./Saves/Models/CAMResNet50.pt")
        CAMResNet.load_state_dict(state_dict)
    else:
        state_dict = torch.load("./Saves/Models/CAMResNet50.pt",
                                map_location=torch.device('cpu'))
        CAMResNet.load_state_dict(state_dict)
    CAMResNet.to(device)
    CAMResNet.eval()

    CAMVGG1 = nn.Sequential(*list(modelVGG.children())[:-2])
    CAMVGG2 = CAMNet(latent_dim=512).to(device)
    CAMVGG = nn.Sequential(CAMVGG1, CAMVGG2)
    if use_gpu:
        state_dict = torch.load("./Saves/Models/CAMVGG19.pt")
        CAMVGG.load_state_dict(state_dict)
    else:
        state_dict = torch.load("./Saves/Models/CAMVGG19.pt",
                                map_location=torch.device('cpu'))
        CAMVGG.load_state_dict(state_dict)
    CAMVGG.to(device)
    CAMVGG.eval()

    images, labels, targets = loadImages(imgfiles, imglabels, imgtargets, input_dir)

    # save_images() saves the original, adversarial example, perturbation,
    # adv. example with perturbed pixels highlighted in red, and a perturbation
    # mask for every image loaded before

    x_cam = CAM(CAMVGG, CAMVGG1, images.clone().to(device), labels.to(device)).unsqueeze(1).cpu()

    # untargeted VGG19 examples
    with open('./attackParams.json', 'r') as f:
        params = json.load(f)["GSE"]["untargeted"]["ImageNet"]
    x_adv = GSEAttack(modelVGG, targeted=False, **params)(images, labels)
    save_images(x_adv, x_cam, images, output_dir + 'VGG19untargeted/')

    # targeted VGG19 examples
    with open('./attackParams.json', 'r') as f:
        params = json.load(f)["GSE"]["targeted"]["ImageNet"]
    x_adv = GSEAttack(modelVGG, targeted=True, **params)(images, targets)
    save_images(x_adv, x_cam, images, output_dir + 'VGG19targeted/')

    x_cam = CAM(CAMResNet, CAMResNet1, images.clone().to(device), labels.to(device)).unsqueeze(1).cpu()

    # untargeted ResNet50 examples
    with open('./attackParams.json', 'r') as f:
        params = json.load(f)["GSE"]["untargeted"]["ImageNet"]
    x_adv = GSEAttack(modelResNet, targeted=False, **params)(images, labels)
    save_images(x_adv, x_cam, images, output_dir + 'ResNet50untargeted/GSE/')

    with open('./attackParams.json', 'r') as f:
        params = json.load(f)["StrAttack"]["untargeted"]["ImageNet"]
    x_adv = StrAttack(modelResNet, targeted=False, **params)(images, labels)
    save_images(x_adv, x_cam, images, output_dir + 'ResNet50untargeted/StrAttack/')

    with open('./attackParams.json', 'r') as f:
        params = json.load(f)["FWnucl"]["untargeted"]["ImageNet"]
    x_adv = FWnucl(modelResNet, targeted=False, **params)(images, labels)
    save_images(x_adv, x_cam, images, output_dir + 'ResNet50untargeted/FWnucl/')
