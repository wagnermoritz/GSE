import torch
import torchvision
from torchvision import transforms, datasets
import argparse
import pandas as pd
import json
import dill

from attacks import *
from utils import *
from models import *

IMAGENET_PATH = "path_to_imagenet_dataset"
NIPS_PATH = "path_to_nips2017_dataset"
CIFAR_PATH = "path_to_cifar10_dataset"
ADV_MODEL_PATH = "path_to_adv_trained_resnet50_weights" # https://github.com/MadryLab/robustness/tree/master
RN20_PATH = "path_to_resnet20_weights"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['ImageNet', 'NIPS2017', 'CIFAR10'], type=str, required=True)
    parser.add_argument('--model', choices=['ResNet50', 'VGG19', 'ViT_B_16', 'ResNet20', 'ResNet50_adv'],
                        help=('Choices ResNet50, VGG19, ResNet50_adv, and ViT_B_16 for ImageNet or NIPS2017 datasets.'
                        'Choice ResNet20 for CIFAR10 dataset.'), type=str, required=True)
    parser.add_argument('--numchunks', default=1, type=int, help=('Number of chunks to split the dataset into.'
                        'The test will be performed on the chunk with the index given by --chunk.'))
    parser.add_argument('--chunk', default=0, type=int, help='Chunk of the dataset the test should be performed on.')
    parser.add_argument('--batchsize', type=int, required=True)
    parser.add_argument('--attack', choices=['GSE', 'FWnucl', 'Homotopy', 'StrAttack', 'SAPF', 'SparseRS', 'PGD0'],
                        type=str, required=True, help='The attack to be tested.')
    parser.add_argument('--targeted', type=int, choices=[0, 1], required=True, help='Test targeted (1) or untargeted (0) attack.')
    parser.add_argument('--sequential', type=int, choices=[0, 1], default=0, help='Run the attack sequentially for every image '
                        'in a batch (1) or not (0). This option is used for comparing the computation times of GSE and SAPF and Homotopy '
                        'attack. GSE and SAPF are the only attacks for which this option can be 1.')

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    if args.dataset in ['ImageNet', 'NIPS2017'] and args.model in ['ResNet20']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataset}.")
    elif args.dataset == 'CIFAR10' and args.model in ['ResNet50', 'VGG19', 'ViT_B_16']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataset}.")
    if args.numchunks <= args.chunk:
        raise ValueError(f"Can't use dataset chunk {args.chunk} when dataset is split into {args.numchunks} chunks."
                        "(indices start at 0)")
    if args.sequential == 1 and args.attack not in ['GSE', 'SAPF']:
        raise ValueError(f"Can only use --sequential=1 with GSE or SAPF.")

    # ------------------------------- datasets -------------------------------
    if args.model == 'ViT_B_16':
        resize = (224, 224)
    else:
        resize = (256, 256)
    
    if args.dataset == 'ImageNet':
        trf = transforms.Compose([transforms.Resize(resize, antialias=None),
                                transforms.ToTensor(),
                                transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        
        # random 10k indices from text file for reproducability
        with open("random_10k_indices.txt", "r") as f:
            string = f.readline()
        idxs = torch.chunk(torch.tensor([int(i) for i in string.split(", ")]), args.numchunks)[args.chunk]

        ImgNetSub = torch.utils.data.Subset(torchvision.datasets.ImageNet(IMAGENET_PATH, split="val", transform=trf), idxs)
        dataloader = torch.utils.data.DataLoader(ImgNetSub, batch_size=args.batchsize)
        numclasses = 1000
        # 10 randomly chosen but fixed label offsets for targeted attack tests
        offsets = [857, 477, 974, 591, 150, 547, 261, 86, 151, 610]

    elif args.dataset == 'NIPS2017':
        trf = transforms.Compose([transforms.Resize(resize, antialias=None),
                                transforms.ToTensor(),
                                transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        
        idxs = torch.chunk(torch.arange(0, 1000), args.numchunks)[args.chunk]
        NIPSlabels = pd.read_csv(NIPS_PATH + 'images.csv')
        NIPSdataset = torch.utils.data.Subset(CustomDataSet(NIPS_PATH + 'images', transform=trf, labels=NIPSlabels), idxs)
        dataloader = torch.utils.data.DataLoader(NIPSdataset, batch_size=args.batchsize)
        numclasses = 1000
        # 10 randomly chosen but fixed label offsets for targeted attack tests
        offsets = [857, 477, 974, 591, 150, 547, 261, 86, 151, 610]

    elif args.dataset == 'CIFAR10':
        trf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        idxs = torch.chunk(torch.arange(0, 10000), args.numchunks)[args.chunk]
        CIFARdataset = torch.utils.data.Subset(datasets.CIFAR10(root=CIFAR_PATH, train=False, download=True, transform=trf), idxs)
        dataloader = torch.utils.data.DataLoader(CIFARdataset, batch_size=args.batchsize)
        numclasses = 10
        # all 9 label offsets for targeted attack tests on CIFAR10
        offsets = [1,2,3,4,5,6,7,8,9]

    # -------------------------------- models --------------------------------
    if args.model == 'ResNet50':
        model = torchvision.models.resnet50(weights='DEFAULT')
    elif args.model == 'VGG19':
        model = torchvision.models.vgg19(weights='DEFAULT')
    elif args.model == 'ViT_B_16':
        model = torchvision.models.vit_b_16(weights='DEFAULT')
    elif args.model == 'ResNet20':
        model = ResNet20()
        state_dict = torch.load(RN20_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
    elif args.model == 'ResNet50_adv':
        model = torchvision.models.resnet50()
        state_dict = torch.load(ADV_MODEL_PATH, map_location='cpu', pickle_module=dill)
        state_dict = {key[len('module.model.'):]: state_dict['model'][key] for key in list(state_dict['model'].keys())[2:] if 'attacker' not in key}
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    # ------------------------------- attacks --------------------------------
    attacks = {'GSE': GSEAttack, 'FWnucl': FWnucl, 'Homotopy': HomotopyAttack, 'StrAttack': StrAttack,
               'SAPF': SAPF, 'SparseRS': SparseRS, 'PGD0': PGD0}
    attack = attacks[args.attack]

    jsonDS = "ImageNet" if args.dataset == "ImageNet" or args.dataset == "NIPS2017" else "CIFAR10"
    jsonT = "targeted" if args.targeted else "untargeted"
    with open('./attackParams.json', 'r') as f:
        params = json.load(f)[args.attack][jsonT][jsonDS]
    if args.attack == 'GSE' and args.model == 'ResNet50_adv':
        params['mu'] = 0.0005
        params['q'] = 0.9
        params['sigma'] = 0.75
        params['k_hat'] = 150

    seq = ''
    if args.sequential:
        params['sequential'] = True
        seq = '_sequential'

    # -------------------------------- test ----------------------------------
    resdir = f'./Outputs/{args.attack}{seq}_{jsonT}_{args.dataset}_{args.model}/'
    os.makedirs(resdir, exist_ok=True)
    torch.manual_seed(0)

    if args.targeted:
        results = test_targeted(attack(model, targeted=True, **params), dataloader, labeloffsets=offsets, numclasses=numclasses)
        if results:
            write_targeted_results(results, resdir + f'{args.numchunks}_{args.chunk}')
        else:
            with open(resdir + f'{args.numchunks}_{args.chunk}' + '_no_adversarial_example_found.txt', 'w') as f:
                f.write('no_adversarial_example_found')
    else:
        results = test_untargeted(attack(model, targeted=False, **params), dataloader)
        if results:
            write_untargeted_results(results, resdir + f'{args.numchunks}_{args.chunk}')
        else:
            with open(resdir + f'{args.numchunks}_{args.chunk}' + '_no_adversarial_example_found.txt', 'w') as f:
                f.write('no_adversarial_example_found')
