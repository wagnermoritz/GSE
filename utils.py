import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
import random
import math
from PIL import Image
import os
import natsort
import pandas as pd
import matplotlib.pyplot as plt


class CustomDataSet(Dataset):
    '''
    Dataset class for NIPS2017 images.
    '''
    def __init__(self, main_dir, transform, labels):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.labels = labels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.labels['ImageId'][idx]) + '.png'
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return (tensor_image, self.labels['TrueLabel'][idx] - 1)
    

def getNIPSDataloader(data_path, b_size):
    '''
    Returns a dataloader for the NIPS2017 dataset using a CustomDataSet object.
    '''
    imagenet_transform = transforms.Compose([transforms.Resize(256, antialias=None),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.5, .5, .5],
                                                            [.5, .5, .5])])

    NIPSlabels = pd.read_csv(data_path + 'images.csv')
    NIPSdataset = CustomDataSet(data_path + 'images',
                                transform=imagenet_transform,
                                labels=NIPSlabels)

    dataloader = torch.utils.data.DataLoader(NIPSdataset, batch_size=b_size,
                                             shuffle=True)
    
    return dataloader

    
def getCIFARDataloader(data_path, b_size, download=False):
    '''
    Returns a dataloader for the CIFAR10 dataset.
    '''
    normalize = transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_path, train=False, download=download,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       normalize])),
        batch_size=b_size, shuffle=True, num_workers=2, pin_memory=True)
    
    return dataloader


def countClusters(pertmask):
    '''
    Counts the number of non-zero pixel clusters in a given perturbation.
    '''
    notdiscovered = torch.zeros((pertmask.size(0) + 2, pertmask.size(1) + 2))
    notdiscovered[1:pertmask.size(0)+1, 1:pertmask.size(1)+1] = pertmask.clone()
    intmask = torch.zeros_like(notdiscovered)
    i = 1
    while notdiscovered.any():
        notdiscovered, intmask = DFS(notdiscovered, intmask, i)
        i += 1
    return intmask.int()


def DFS(notdiscovered, intmask, i):
    '''
    Performs DFS on a perturbation by treating adjacent non-zero pixels as
    neighboring nodes of a graph.
    '''
    neighbors = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    nonzeroidxs = torch.nonzero(notdiscovered, as_tuple=True)
    rnd = random.randint(0, len(nonzeroidxs[0])-1)
    start = [idx[rnd] for idx in nonzeroidxs]
    stack = [start]

    while stack:
        v = stack.pop()
        if notdiscovered[v]:
            notdiscovered[v] = 0
            intmask[v] = i
            for offset in neighbors:
                if notdiscovered[v[0] + offset[0], v[1] + offset[1]]:
                    stack.append((v[0] + offset[0], v[1] + offset[1]))
    return notdiscovered, intmask


def ASM(model, x, t, t0, device):
    '''
    Computes the adversarial saliency map of the given model at x wrt target
    label t and true label t0.
    '''
    x.requires_grad = True
    out = model(x)
    grd = torch.zeros(out.shape, device=device)
    grd[:, t] = 1
    out.backward(gradient=grd, retain_graph=True)
    dZ_t_dx = x.grad.clone()
    x.grad.zero_()
    grd = torch.zeros(out.shape, device=device)
    grd[:, t0] = 1
    out.backward(gradient=grd)
    dZ_t0_dx = x.grad.clone()
    x.detach_()
    cond = torch.zeros_like(x).bool()
    return torch.where(cond, torch.zeros_like(x), dZ_t_dx * dZ_t0_dx.abs())


def ASM_percentile(asm, P, device):
    '''
    Returns a list of Pth percentiles for all adversarial saliency maps in the
    batch asm.
    '''
    asm = asm.view(asm.size(0), -1)
    asm = torch.sort(asm, dim=-1)[0]
    out = []
    for a in asm:
        a = torch.cat([torch.zeros((1,), device=device), a[a.nonzero()[:, 0]]])
        ordrank = math.ceil(P / 100 * a.size(0))
        out.append(a[ordrank])
    return torch.tensor(out, device=device)


def IS(model, x, y, delta, P, t, device):
    '''
    Computes the interpretability scores for the batch of perturbations delta.
    '''
    mask = torch.norm(delta, p=2, dim=(1, 2, 3)) > 0
    x = x[mask]
    y = y[mask]
    delta = delta[mask]
    x_asm = ASM(model, x, t, y, device)
    perc = ASM_percentile(x_asm, P, device)[:, None, None, None]
    x_asm = torch.where(x_asm > perc, torch.ones_like(x_asm), torch.zeros_like(x_asm))
    return torch.norm(x_asm * delta, p=2, dim=(1, 2, 3)) / torch.norm(delta, p=2, dim=(1, 2, 3))


class CAMNet(nn.Module):
    '''
    Class for the last fully connected layer of CNNs used for computing the
    class activation map.
    '''
    def __init__(self, numclasses=1000, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, numclasses, bias=False)

    def forward(self, x):
        sh = x.shape
        x = x.view(*sh[:2], sh[2] * sh[3]).mean(-1).view(sh[0], -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


def CAM(model1, model2, x, y):
    '''
    Computes the class activation map of the CNN model2 with the last fully
    connected layers removed.
    '''
    transf = transforms.Compose([transforms.Resize((256, 256), torchvision.transforms.InterpolationMode.BICUBIC)])
    weight = list(model1.parameters())[-1].data
    last_conv = model2(x).detach()
    sh = last_conv.shape
    cam = last_conv.reshape(*sh[:2], sh[2] * sh[3])
    cam = (weight[y][:, :, None] * cam).sum(1)
    cam = cam.reshape(sh[0], *sh[2:])
    cam -= cam.min()
    cam /= cam.max()
    cam = transf(cam)
    return cam


def loadImages(image_files, labels, targets, path_to_images):
    '''
    Loads images from the paths given in the list of image files and returns the
    images, labels, and targets as tensors.
    '''
    labels = torch.tensor(labels).long()
    targets = torch.tensor(targets).long()
    images = []
    trf = transforms.Compose([transforms.Resize((256, 256), antialias=None),
                            transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
    for imf in image_files:
        x = Image.open(path_to_images + imf)
        x = transforms.ToTensor()(x)
        x = trf(x)
        images.append(x.clone())

    images = torch.stack(images)

    return images, labels, targets


def extract_patches(x, size=8):
    '''
    Extracts all n by n pixel patches from every image in the batch x.
    '''
    B, C, H, W = x.shape

    kernel = torch.zeros((size ** 2, size ** 2))
    kernel[range(size**2), range(size**2)] = 1.0
    kernel = kernel.view(size**2, 1, size, size)
    kernel = kernel.repeat(C, 1, 1, 1).to(x.device)

    out = F.conv2d(x, kernel, groups=C)
    out = out.view(B, C, size, size, -1)
    out = out.permute(0, 4, 1, 2, 3)
    return out.contiguous()


def d_2_0(x, size=8):
    '''
    Computed the value of the d_2_0 function, i.e. the number of non-zero
    n by n patches in each image in the batch x.
    '''
    l20s = []
    for x_ in x:
        patches = extract_patches(x_.unsqueeze(0), size)
        l2s = torch.norm(patches, p=2, dim=(2, 3, 4))
        l20s.append((l2s != 0).float().sum().item())
    return torch.tensor(l20s)


def test_targeted(attack, dataloader, labeloffsets, numclasses, num_batches):
    '''
    Evaluates a given targeted attack in terms of ASR, sparsity, number of
    clusters, 2-norm, d_2_0 value, each for best, average, and worst case, and
    interpretability scores and computation time.
    '''
    percentiles = [50, 60, 70, 80, 90]
    IS_scores = [[] for _ in percentiles]
    l0s3 = [[], [], []]
    l2s3 = [[], [], []]
    l20s3 = [[], [], []]
    clusters3 = [[], [], []]
    successes3 = [[], [], []]
    total_time = 0
    n = 0
    device = attack.device

    if num_batches is None:
        num_batches = len(dataloader)

    for i, (batch, labels) in enumerate(dataloader):
        if i == num_batches:
            break
        mask = (torch.argmax(attack.model(batch.to(device)), dim=1).cpu() == labels)
        batch = batch[mask]
        labels = labels[mask]

        l0s = []
        l2s = []
        l20s = []
        clusters = []
        successes = []

        for j, offs in enumerate(labeloffsets):
            print(f'Batch {i+1}/{num_batches}, Label {j+1}/{len(labeloffsets)}')
            x = batch.clone().to(device)
            y = labels.clone().to(device)

            n += x.size(0)
            before = time.perf_counter()
            x_adv = attack(x, ((y + offs) % numclasses).long()).to(device)
            after = time.perf_counter()
            total_time += after - before

            mask = (torch.argmax(attack.model(x_adv), dim=1) == (y+offs)%numclasses)

            successes.append(mask)
            mask = torch.logical_not(mask)

            l0s.append(torch.norm((x_adv - x).abs().mean(1), p=0, dim=(1,2)).cpu())
            l2s.append(torch.norm((x_adv - x), p=2, dim=(1,2,3)).cpu())
            l20s.append(d_2_0(x_adv - x).cpu())
            l0s[-1][mask] = 1e10
            l2s[-1][mask] = 1e10
            l20s[-1][mask] = 1e10
            clusters.append(torch.tensor([countClusters((x_adv.cpu() - x.cpu())[idx].abs().mean(0)!=0).max().int().item() for idx in range(len(x))]))
            clusters[-1][mask] = 1e10

            for j, P in enumerate(percentiles):
                IS_scores[j].append(IS(attack.model, x.detach(), y.detach(), (x_adv - x).detach(), P, (y+offs)%numclasses, device))

        Tsucc = torch.stack(successes, dim=0).cpu()
        Tbestsucc = torch.max(Tsucc, dim=0)[0]
        Tworstsucc = torch.min(Tsucc, dim=0)[0]
        successes3[0].append(Tbestsucc)
        successes3[1].append(Tsucc.float())
        successes3[2].append(Tworstsucc)

        def getCases(t):
            T = torch.stack(t, dim=0)
            Tbest = torch.min(T, dim=0)[0]
            T = torch.where(T > 1e9, torch.full_like(T, -1e10), T)
            Tworst = torch.max(T, dim=0)[0]
            mask = T > -1e9
            T = T[torch.logical_and(mask, successes3[1][-1])]
            mask = Tbest < 1e9
            Tbest = Tbest[torch.logical_and(mask, successes3[0][-1])]
            Tworst = Tworst[torch.logical_and(mask, successes3[2][-1])]

            return Tbest, T.float(), Tworst

        tmp = getCases(l0s)
        if len(tmp[0]):
            l0s3[0].append(tmp[0])
        l0s3[1].append(tmp[1])
        if len(tmp[2]):
            l0s3[2].append(tmp[2])

        tmp = getCases(l2s)
        if len(tmp[0]):
            l2s3[0].append(tmp[0])
        l2s3[1].append(tmp[1])
        if len(tmp[2]):
            l2s3[2].append(tmp[2])

        tmp = getCases(l20s)
        if len(tmp[0]):
            l20s3[0].append(tmp[0])
        l20s3[1].append(tmp[1])
        if len(tmp[2]):
            l20s3[2].append(tmp[2])

        tmp = getCases(clusters)
        if len(tmp[0]):
            clusters3[0].append(tmp[0])
        clusters3[1].append(tmp[1])
        if len(tmp[2]):
            clusters3[2].append(tmp[2])

    IS_scores = [torch.cat(score) for score in IS_scores]

    for i in range(3):
        if not len(l0s3[i]):
            l0s3[i].append(torch.tensor([torch.nan]))
            l2s3[i].append(torch.tensor([torch.nan]))
            l20s3[i].append(torch.tensor([torch.nan]))
            clusters3[i].append(torch.tensor([torch.nan]))

    return l0s3, clusters3, successes3, total_time / n, l2s3, l20s3, IS_scores


def test_untargeted(attack, dataloader, num_batches):
    '''
    Evaluates a given untargeted attack in terms of ASR, sparsity, number of
    clusters, 2-norm, d_2_0 value, and computation time.
    '''
    l0s = []
    l2s = []
    l20s = []
    clusters = []
    successes = []
    total_time = 0
    n = 0
    device = attack.device

    for i, (x, y) in enumerate(dataloader):
        if i == num_batches:
            break
        print(f'Batch {i+1}/{num_batches}')
        x = x.to(device)
        y = y.to(device)

        mask = (torch.argmax(attack.model(x), dim=1) == y)
        x = x[mask]
        y = y[mask]
        n += x.size(0)

        before = time.perf_counter()
        x_adv = attack(x, y).to(device)
        after = time.perf_counter()

        mask = (torch.argmax(attack.model(x_adv), dim=1) != y)

        if not mask.any():
            continue

        successes.append(mask)
        x = x[mask]
        x_adv = x_adv[mask]

        total_time += after - before
        l0s.append(torch.norm((x_adv - x).abs().mean(1), p=0, dim=(1,2)).cpu())
        l2s.append(torch.norm((x_adv - x), p=2, dim=(1,2,3)).cpu())
        l20s.append(d_2_0(x_adv - x).cpu())
        clusters.append(torch.tensor([countClusters((x_adv.cpu() - x.cpu())[idx].abs().mean(0)!=0).max().int().item() for idx in range(len(x))]))

    return l0s, clusters, successes, total_time / n, l2s, l20s


def save_images(x_adv, x_cam, images, dir_str):
    '''
    Saves images to the path dir_str.
    '''
    x_adv = x_adv.cpu()
    images = images.cpu()
    x_cam = x_cam.cpu()
    os.makedirs(dir_str, exist_ok=True)

    for i in range(len(images)):
        fig = plt.figure(figsize=(6,6), dpi=200)
        plt.imshow(x_adv[i].squeeze().permute(1,2,0) * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig(dir_str + f'{i+1}_adv_ex.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

        fig = plt.figure(figsize=(6,6), dpi=200)
        plt.imshow((x_adv[i].squeeze() - images[i]).permute(1,2,0) * 4 + 0.5)
        plt.axis('off')
        plt.savefig(dir_str + f'{i+1}_pert.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

        fig = plt.figure(figsize=(6,6), dpi=200)
        plt.imshow(((x_adv[i].squeeze() - images[i]).mean(0, keepdim=True).permute(1,2,0) != 0).float(), cmap='gray')
        plt.axis('off')
        plt.savefig(dir_str + f'{i+1}_pertmask.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

        fig = plt.figure(figsize=(6,6), dpi=200)
        plt.imshow(images[i].permute(1,2,0) * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig(dir_str + f'{i+1}_original.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

        fig = plt.figure(figsize=(6,6), dpi=200)
        plt.imshow(x_cam[i].squeeze() * 10 + 0.5, cmap='jet_r')
        plt.axis('off')
        plt.savefig(dir_str + f'{i+1}_CAM.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)


def write_untargeted_results(results, file):
    '''
    Writes results in a file.
    '''
    string = f"L0: {torch.cat(results[0]).mean().item()}\n"
    string += f"L2: {torch.cat(results[5]).mean().item()}\n"
    string += f"L2,0: {torch.cat(results[6]).mean().item()}\n"
    string += f"clusters: {torch.cat(results[1]).float().mean().item()}\n"
    string += f"GB clusters: {torch.cat(results[2]).float().mean().item()}\n"
    string += f"ASR: {torch.cat(results[3]).float().mean().item()}\n"
    string += f"time: {results[4]}"

    with open(file, 'w') as f:
        f.write(string)


def write_targeted_results(results, file):
    '''
    Writes results in a file.
    '''
    string = f"L0: best: {torch.cat(results[0][0]).mean().item()}, avg: {torch.cat(results[0][1]).mean().item()}, worst: {torch.cat(results[0][2]).mean().item()}\n"
    string += f"L2: best: {torch.cat(results[5][0]).mean().item()}, avg: {torch.cat(results[5][1]).mean().item()}, worst: {torch.cat(results[5][2]).mean().item()}\n"
    string += f"L2,0: best: {torch.cat(results[6][0]).mean().item()}, avg: {torch.cat(results[6][1]).mean().item()}, worst: {torch.cat(results[6][2]).mean().item()}\n"
    string += f"Clusters: best: {torch.cat(results[1][0]).float().mean().item()}, avg: {torch.cat(results[1][1]).float().mean().item()}, worst: {torch.cat(results[1][2]).float().mean().item()}\n"
    string += f"GB Clusters: best: {torch.cat(results[2][0]).float().mean().item()}, avg: {torch.cat(results[2][1]).float().mean().item()}, worst: {torch.cat(results[2][2]).float().mean().item()}\n"
    string += f"ASR: best: {torch.cat(results[3][0]).float().mean().item()}, avg: {torch.cat(results[3][1]).float().mean().item()}, worst: {torch.cat(results[3][2]).float().mean().item()}\n"
    string += f"Time: best: {results[4]}\n\n"

    string += "Percentile vs interpretability score:\nP, IS\n"
    for p, ISres in zip([30, 40, 50, 60, 70, 80, 90], results[7]):
        string += f"{p}, {ISres.mean().item()}\n"

    with open(file, 'w') as f:
        f.write(string)
