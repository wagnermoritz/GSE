import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import os

# https://github.com/akamaster/pytorch_resnet_cifar10
from pytorch_resnet_cifar10.resnet import resnet20
from attacks import * 
from utils import *


dir_str = 'path to directory'
# info for figures 1, 3, 4
imgfiles = ['list of image file names'] # in dir_str + 'Saves/Data/images/'
imglabels = ['list of labels as ints']
imgtargets = ['list of targets as ints']

if __name__ == "__main__":

    ################################# SETUP ###################################
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    modelResNet = torchvision.models.resnet50(weights='DEFAULT')
    modelVGG = torchvision.models.vgg19(weights='DEFAULT')
    modelResNet.eval().to(device)
    modelVGG.eval().to(device)

    modelCIFAR = resnet20()
    if use_gpu:
        state_dict = torch.load(dir_str + "Saves/Models/CIFARModel.pt")
        modelCIFAR.load_state_dict(state_dict)
    else:
        state_dict = torch.load(dir_str + "Saves/Models/CIFARModel.pt",
                                map_location=torch.device('cpu'))
        modelCIFAR.load_state_dict(state_dict)
    modelCIFAR.to(device)
    modelCIFAR.eval()

    CAMResNet1 = nn.Sequential(*list(modelResNet.children())[:-2])
    CAMResNet2 = CAMNet(latent_dim=2048).to(device)
    CAMResNet = nn.Sequential(CAMResNet1, CAMResNet2)
    if use_gpu:
        state_dict = torch.load(dir_str + "Saves/Models/CAMResNet50.pt")
        CAMResNet.load_state_dict(state_dict)
    else:
        state_dict = torch.load(dir_str + "Saves/Models/CAMResNet50.pt",
                                map_location=torch.device('cpu'))
        CAMResNet.load_state_dict(state_dict)
    CAMResNet.to(device)
    CAMResNet.eval()

    CAMVGG1 = nn.Sequential(*list(modelVGG.children())[:-2])
    CAMVGG2 = CAMNet(latent_dim=512).to(device)
    CAMVGG = nn.Sequential(CAMVGG1, CAMVGG2)
    if use_gpu:
        state_dict = torch.load(dir_str + "Saves/Models/CAMVGG19.pt")
        CAMVGG.load_state_dict(state_dict)
    else:
        state_dict = torch.load(dir_str + "Saves/Models/CAMVGG19.pt",
                                map_location=torch.device('cpu'))
        CAMVGG.load_state_dict(state_dict)
    CAMVGG.to(device)
    CAMVGG.eval()
    
    ############################# Figures 1, 2, 3 #############################
    print('Working on data for figures 1, 2, and 3.')
    images, labels, targets = loadImages(imgfiles, imglabels, imgtargets, dir_str + 'Saves/Data/images/')

    # save_images() saves the original, adversarial example, perturbation,
    # adv. example with perturbed pixels highlighted in red, and a perturbation
    # mask for every image loaded before
    # untargeted VGG19 examples
    x_cam = CAM(CAMVGG, CAMVGG1, images.clone().to(device), labels.to(device)).unsqueeze(1).cpu()
    att = GSEAttack(modelVGG, targeted=False, img_range=(-1, 1), mu=0.1, beta=0.025, k_hat=50, q=0.9)
    x_adv = att(images, labels)
    save_images(x_adv, x_cam, images, dir_str + 'Outputs/VGG19untargeted/')
    # targeted VGG19 examples
    att = GSEAttack(modelVGG, targeted=True, img_range=(-1, 1), mu=0.1, beta=0.025, k_hat=50, q=0.9)
    x_adv = att(images, targets)
    save_images(x_adv, x_cam, images, dir_str + 'Outputs/VGG19targeted/')
    # untargeted ResNet50 examples
    x_cam = CAM(CAMResNet, CAMResNet1, images.clone().to(device), labels.to(device)).unsqueeze(1).cpu()
    att = GSEAttack(modelResNet, targeted=False, img_range=(-1, 1), mu=0.1, beta=0.025, k_hat=50, q=0.9)
    x_adv = att(images, labels)
    save_images(x_adv, x_cam, images, dir_str + 'Outputs/ResNet50untargeted/')
    # targeted ResNet50 examples
    att = GSEAttack(modelResNet, targeted=True, img_range=(-1, 1), mu=0.1, beta=0.025, k_hat=50, q=0.9)
    x_adv = att(images, targets)
    save_images(x_adv, x_cam, images, dir_str + 'Outputs/ResNet50targeted/')

    print('done')

    ################################# Table 1 #################################
    print('Working on data for table 1.')
    # untargeted
    dataloader = getCIFARDataloader(dir_str + 'Saves/Data/CIFAR10/', 100, download=True)
    results = test_untargeted(HomotopyAttack(modelCIFAR, dec_factor=0.9, val_c=1e-1, val_w1=1e-3,
                                             val_w2=1e-5, val_gamma=0.96, beta=1e-1, iter_init=50,
                                             iter_inc=[100, 200, 300, 400, 500], n_segments=100),
                              dataloader, num_batches=1)
    write_untargeted_results(results, dir_str + 'Outputs/Table1/Homotopy_untargeted.txt')

    results = test_untargeted(GSEAttack(modelCIFAR, targeted=False, sequential=True, mu=1, beta=0.0025, k_hat=30, q=0.25),
                              dataloader, num_batches=1)
    write_untargeted_results(results, dir_str + 'Outputs/Table1/GSE_untargeted.txt')

    # targeted
    dataloader = getCIFARDataloader(dir_str + 'Saves/Data/', 11)
    results = test_targeted(HomotopyAttack(modelCIFAR, targeted=True, dec_factor=0.8, val_c=1e-1, val_w1=1e-3,
                                           val_w2=1e-5, val_gamma=0.96, beta=1e-1, iter_init=50,
                                           iter_inc=[100, 200, 300, 400, 500], n_segments=100),
                            dataloader, num_batches=1, labeloffsets=[1,2,3,4,5,6,7,8,9], numclasses=10)
    write_targeted_results(results, dir_str + 'Outputs/Table1/Homotopy_targeted.txt')

    results = test_targeted(GSEAttack(modelCIFAR, targeted=True, sequential=True, mu=1, beta=0.0025, k_hat=30, q=0.25),
                            dataloader, num_batches=1, labeloffsets=[1,2,3,4,5,6,7,8,9], numclasses=10)
    write_targeted_results(results, dir_str + 'Outputs/Table1/GSE_targeted.txt')

    print('done')

    ################################# Table 2 #################################
    print('Working on data for table 2.')
    os.makedirs(dir_str + 'Outputs/Table2/', exist_ok=True)
    # CIFAR10
    dataloader = getCIFARDataloader(dir_str + 'Saves/Data/CIFAR10/', 500)
    results = test_untargeted(GSEAttack(modelCIFAR, targeted=False, mu=1, beta=0.0025, k_hat=30, q=0.25),
                              dataloader, num_batches=2)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/GSE_CIFAR10.txt')

    results = test_untargeted(StrAttack(modelCIFAR, img_range=(-1, 1), targeted=False, c=0.25),
                              dataloader, num_batches=2)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/StrAttack_CIFAR10.txt')
    
    results = test_untargeted(FWnucl(modelCIFAR, img_range=(-1, 1), targeted=False),
                              dataloader, num_batches=2)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/FWnucl_CIFAR10.txt')

    # ImageNet VGG19
    dataloader = getNIPSDataloader(dir_str + 'Saves/Data/NIPS2017/', 50)
    results = test_untargeted(GSEAttack(modelVGG, targeted=False, mu=0.1, beta=0.025, k_hat=50, q=0.95),
                              dataloader, num_batches=20)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/GSE_NIPS2017VGG.txt')

    results = test_untargeted(StrAttack(modelVGG, img_range=(-1, 1), targeted=False, c=2.5),
                              dataloader, num_batches=20)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/StrAttack_NIPS2017VGG.txt')
    
    results = test_untargeted(FWnucl(modelVGG, img_range=(-1, 1), targeted=False),
                              dataloader, num_batches=20)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/FWnucl_NIPS2017VGG.txt')

    # ImageNet ResNet50
    results = test_untargeted(GSEAttack(modelResNet, targeted=False, mu=0.1, beta=0.025, k_hat=50, q=0.95),
                              dataloader, num_batches=20)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/GSE_NIPS2017ResNet.txt')

    results = test_untargeted(StrAttack(modelResNet, img_range=(-1, 1), targeted=False, c=2.5),
                              dataloader, num_batches=20)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/StrAttack_NIPS2017ResNet.txt')
    
    results = test_untargeted(FWnucl(modelResNet, img_range=(-1, 1), targeted=False),
                              dataloader, num_batches=20)
    write_untargeted_results(results, dir_str + 'Outputs/Table2/FWnucl_NIPS2017ResNet.txt')

    print('done')

    ########################## Table 3 and Figure 4 ###########################
    print('Working on data for table 1 and figure 4.')
    os.makedirs(dir_str + 'Outputs/Table3Fig4/', exist_ok=True)
    # CIFAR10
    dataloader = getCIFARDataloader(dir_str + 'Saves/Data/CIFAR10/', 111)
    results = test_targeted(GSEAttack(modelCIFAR, targeted=False, mu=1, beta=0.0025, k_hat=30, q=0.25),
                            dataloader, labeloffsets=[1,2,3,4,5,6,7,8,9], numclasses=10, num_batches=1)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/GSE_CIFAR10.txt')

    results = test_targeted(StrAttack(modelCIFAR, img_range=(-1, 1), targeted=False, c=0.25),
                            dataloader, labeloffsets=[1,2,3,4,5,6,7,8,9], numclasses=10, num_batches=1)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/StrAttack_CIFAR10.txt')
    
    results = test_targeted(FWnucl(modelCIFAR, img_range=(-1, 1), targeted=False),
                            dataloader, labeloffsets=[1,2,3,4,5,6,7,8,9], numclasses=10, num_batches=1)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/FWnucl_CIFAR10.txt')

    # ImageNet VGG19
    dataloader = getNIPSDataloader(dir_str + 'Saves/Data/NIPS2017/', 50)
    offsets = torch.arange(1, 1000)[torch.randperm(999)][:10].tolist()
    results = test_targeted(GSEAttack(modelVGG, targeted=False, mu=0.1, beta=0.025, k_hat=75, q=0.95),
                            dataloader, labeloffsets=offsets, numclasses=1000, num_batches=2)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/GSE_NIPS2017VGG.txt')

    results = test_targeted(StrAttack(modelVGG, img_range=(-1, 1), targeted=False, c=2.5),
                            dataloader, labeloffsets=offsets, numclasses=1000, num_batches=2)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/StrAttack_NIPS2017VGG.txt')
    
    results = test_targeted(FWnucl(modelVGG, img_range=(-1, 1), targeted=False),
                            dataloader, labeloffsets=offsets, numclasses=1000, num_batches=2)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/FWnucl_NIPS2017VGG.txt')

    # ImageNet ResNet50
    results = test_targeted(GSEAttack(modelResNet, targeted=False, mu=0.1, beta=0.03, k_hat=75, q=0.95),
                            dataloader, labeloffsets=offsets, numclasses=1000, num_batches=2)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/GSE_NIPS2017ResNet.txt')

    results = test_targeted(StrAttack(modelResNet, img_range=(-1, 1), targeted=False, c=2.5),
                            dataloader, labeloffsets=offsets, numclasses=1000, num_batches=2)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/StrAttack_NIPS2017ResNet.txt')
    
    results = test_targeted(FWnucl(modelResNet, img_range=(-1, 1), targeted=False),
                            dataloader, labeloffsets=offsets, numclasses=1000, num_batches=2)
    write_targeted_results(results, dir_str + 'Outputs/Table3Fig4/FWnucl_NIPS2017ResNet.txt')

    print('done')