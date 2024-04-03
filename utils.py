import torch
import torch.utils.data
from torch.nn import functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import time
import random
import math
from PIL import Image
import os
import natsort
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
    neighbors = [[-1, 0], [0, -1], [0, 1], [1, 0]]
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
        x = Image.open(path_to_images + imf).convert('RGB')
        x = transforms.ToTensor()(x)
        x = trf(x)
        images.append(x.clone())

    images = torch.stack(images)

    return images, labels, targets


def extract_patches(x, n=8):
    '''
    Extracts all n by n pixel patches from every image in the batch x.
    '''
    B, C, H, W = x.shape

    kernel = torch.zeros((n ** 2, n ** 2))
    kernel[range(n**2), range(n**2)] = 1.0
    kernel = kernel.view(n**2, 1, n, n)
    kernel = kernel.repeat(C, 1, 1, 1).to(x.device)

    out = F.conv2d(x, kernel, groups=C)
    out = out.view(B, C, n, n, -1)
    out = out.permute(0, 4, 1, 2, 3)
    return out.contiguous()


def d_2_0(x, n=8):
    '''
    Computed the value of the d_2_0 function, i.e. the number of non-zero
    n by n patches in each image in the batch x.
    '''
    l20s = []
    for x_ in x:
        patches = extract_patches(x_.unsqueeze(0), n)
        l2s = torch.norm(patches, p=2, dim=(2, 3, 4))
        l20s.append((l2s != 0).float().sum().item())
    return torch.tensor(l20s)


def test_targeted(attack, dataloader, labeloffsets, numclasses, num_batches=None):
    '''
    Evaluates a given targeted attack in terms of ASR, sparsity, number of
    clusters, 2-norm, d_2_0 value (each for best, average, and worst case),
    interpretability scores, and computation time.
    Best, average, and worst case are computed for each image over all the corr.
    targeted adv. examples with targets (target+labeloffsets[i])%numcalsses.
    '''
    percentiles = [50, 60, 70, 80, 90]
    IS_scores = [[] for _ in percentiles]

    # lists for saving best, average, and worst case for every sample
    l0_baw = [[], [], []]
    l2_baw = [[], [], []]
    l20_baw = [[], [], []]
    clusters_baw = [[], [], []]
    successes_baw = [[], [], []]
    total_time = 0
    n = 0
    device = attack.device

    if num_batches is None:
        num_batches = len(dataloader)

    for i, (batch, labels) in enumerate(dataloader):
        if i == num_batches:
            break
        # only consider benign samples that are classified correctly
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
            successes.append(mask.cpu())

            l0s.append(torch.norm((x_adv - x).abs().mean(1), p=0, dim=(1,2)).cpu())
            l2s.append(torch.norm((x_adv - x), p=2, dim=(1,2,3)).cpu())
            l20s.append(d_2_0(x_adv - x).cpu())
            clusters.append(torch.tensor([countClusters((x_adv.cpu() - x.cpu())[idx].abs().mean(0)!=0).max().int().item() for idx in range(len(x))]))

            for j, P in enumerate(percentiles):
                IS_scores[j].append(IS(attack.model, x.detach(), y.detach(), (x_adv - x).detach(), P, (y+offs)%numclasses, device))

        # best, average, and worst case for attack success
        Tsucc = torch.stack(successes, dim=0)
        Tbestsucc = torch.max(Tsucc, dim=0)[0]
        Tworstsucc = torch.min(Tsucc, dim=0)[0]
        successes_baw[0].append(Tbestsucc)
        successes_baw[1].append(Tsucc.float().reshape(-1))
        successes_baw[2].append(Tworstsucc)

        def getCases(t, baw):
            T = torch.stack(t, dim=0)
            # for the best case, take the best successful attack or nothing if
            # there is no successful attack
            Tbest = torch.min(torch.where(Tsucc.bool(), T, torch.full_like(T, 1e9)), dim=0)[0]
            Tbest = Tbest[Tbestsucc.bool()].reshape(-1)
            # for the worst case, take the worst attack or nothing if at least
            # one attack was not successful
            Tworst = torch.max(T, dim=0)[0][Tworstsucc.bool()].reshape(-1)
            T = T.reshape(-1)

            if len(Tbest):
                baw[0].append(Tbest)
            baw[1].append(T)
            if len(Tworst):
                baw[2].append(Tworst)

            return baw
        
        # best, average, worst case for all other metrics depending on
        # the success of the attack
        l0_baw = getCases(l0s, l0_baw)
        l2_baw = getCases(l2s, l2_baw)
        l20_baw = getCases(l20s, l20_baw)
        clusters_baw = getCases(clusters, clusters_baw)

    IS_scores = [torch.cat(score) for score in IS_scores]

    for i in range(3):
        if not len(l0_baw[i]):
            l0_baw[i].append(torch.tensor([torch.nan]))
            l2_baw[i].append(torch.tensor([torch.nan]))
            l20_baw[i].append(torch.tensor([torch.nan]))
            clusters_baw[i].append(torch.tensor([torch.nan]))
    
    return l0_baw, clusters_baw, successes_baw, total_time, l2_baw, l20_baw, IS_scores, n


def test_untargeted(attack, dataloader, num_batches=None):
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

    if num_batches is None:
        num_batches = len(dataloader)

    for i, (x, y) in enumerate(dataloader):
        if i == num_batches:
            break
        print(f'Batch {i+1}/{num_batches}')
        x = x.to(device)
        y = y.to(device)

        # only consider benign samples that are classified correctly
        mask = (torch.argmax(attack.model(x), dim=1) == y)
        x = x[mask]
        y = y[mask]
        n += x.size(0)

        before = time.perf_counter()
        x_adv = attack(x, y).to(device)
        after = time.perf_counter()

        mask = (torch.argmax(attack.model(x_adv), dim=1) != y)
        successes.append(mask)

        if not mask.any():
            continue

        x = x[mask]
        x_adv = x_adv[mask]

        total_time += after - before
        l0s.append(torch.norm((x_adv - x).abs().mean(1), p=0, dim=(1,2)).cpu())
        l2s.append(torch.norm((x_adv - x), p=2, dim=(1,2,3)).cpu())
        l20s.append(d_2_0(x_adv - x).cpu())
        clusters.append(torch.tensor([countClusters((x_adv.cpu() - x.cpu())[idx].abs().mean(0)!=0).max().int().item() for idx in range(len(x))]))

    return l0s, clusters, successes, total_time, l2s, l20s, n


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


def write_untargeted_results(results, filename):
    '''
    Writes results in a file.
    '''
    string = "L0, L2, L2_0, clusters, ASR\n"
    for l0, l2, l20, cl, asr in zip(torch.cat(results[0]), torch.cat(results[4]),
                                    torch.cat(results[5]), torch.cat(results[1]),
                                    torch.cat(results[2])):
        string += f"{l0}, {l2}, {l20}, {cl}, {asr}\n"

    with open(filename + "_results.txt", 'w') as f:
        f.write(string)
    with open(filename + "_time.txt", 'w') as f:
        f.write(f"time: {results[3]}, number of samples: {results[6]}")


def write_targeted_results(results, filename):
    '''
    Writes results in a file.
    '''
    string = "L0, L2, L2_0, clusters, ASR\n"
    for l0, l2, l20, cl, asr in zip(torch.cat(results[0][0]), torch.cat(results[4][0]),
                                    torch.cat(results[5][0]), torch.cat(results[1][0]),
                                    torch.cat(results[2][0])):
        string += f"{l0}, {l2}, {l20}, {cl}, {asr}\n"

    with open(filename + "_results_best.txt", 'w') as f:
        f.write(string)

    string = "L0, L2, L2_0, clusters, ASR\n"
    for l0, l2, l20, cl, asr in zip(torch.cat(results[0][1]), torch.cat(results[4][1]),
                                    torch.cat(results[5][1]), torch.cat(results[1][1]),
                                    torch.cat(results[2][1])):
        string += f"{l0}, {l2}, {l20}, {cl}, {asr}\n"

    with open(filename + "_results_average.txt", 'w') as f:
        f.write(string)

    string = "L0, L2, L2_0, clusters, ASR\n"
    for l0, l2, l20, cl, asr in zip(torch.cat(results[0][2]), torch.cat(results[4][2]),
                                    torch.cat(results[5][2]), torch.cat(results[1][2]),
                                    torch.cat(results[2][2])):
        string += f"{l0}, {l2}, {l20}, {cl}, {asr}\n"

    with open(filename + "_results_worst.txt", 'w') as f:
        f.write(string)

    with open(filename + "_time.txt", 'w') as f:
        f.write(f"time: {results[3]}, number of samples: {results[7]}")


    string = "P50, P60, P70, P80, P90\n"
    for p50, p60, p70, p80, p90 in zip(*results[6]):
        string += f"{p50}, {p60}, {p70}, {p80}, {p90}\n"

    with open(filename + "_IS.txt", 'w') as f:
        f.write(string)
