import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from self_models import fake_vs_real_Model

path = "images"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((300, 300)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
)
data = torchvision.datasets.ImageFolder(root=path, transform=transform)
# print(len(data))
images_loader = DataLoader(data, batch_size=1, shuffle=False, pin_memory=True)
images, _ = next(iter(images_loader))
# print(images)

labels = np.zeros(32, dtype=np.int8)
labels[0:16] = 1
# print(labels.sum())

messup_data = torchvision.datasets.ImageFolder(root="messup_data", transform=transform)
# print(len(data))
messup_images_loader = DataLoader(messup_data, batch_size=16, shuffle=True, pin_memory=True)
messup_images, messup_labels = next(iter(messup_images_loader))
# print(images)

classes = ["fake", "real"]
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def denormalize(image):
    image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
    image = image.permute(1, 2, 0)  # Changing from 3x224x224 to 224x224x3
    image = torch.clamp(image, 0, 1)
    return image


def imshow(imgs, lbs, predict=False, pred_lbs=None, mess_up=False):
    if not predict:
        fig = plt.figure(figsize=(20, 10))
        for i, img in enumerate(imgs):
            img = denormalize(img)
            fig.add_subplot(4, 8, i + 1)
            plt.axis("off")
            plt.title(classes[lbs[i]])
            plt.imshow(img)
    else:
        assert (predict == True), "predicted_labels should not be None."
        fig = plt.figure(figsize=(20, 10))
        if mess_up:
            classes_messup = ["fake_actualreal", "real_actualfake"]
            for i, img in enumerate(imgs):
                img = denormalize(img)
                fig.add_subplot(1, 1, i + 1)
                plt.subplots_adjust(hspace=0.7)
                plt.axis("off")
                plt.title(f"GroundTruth: \n{classes_messup[lbs[i]]}\nPredicted: {classes[pred_lbs[i]]}")
                plt.imshow(img)
                if classes_messup[lbs[i]][-4:] == classes[pred_lbs[i]]:
                    plt.text(0, 330, "Correct", color="g")
                else:
                    plt.text(0, 330, "Wrong", color="r")

            # plt.savefig("predicted_messup.jpg")
        else:
            for i, img in enumerate(imgs):
                img = denormalize(img)
                fig.add_subplot(1, 1, i + 1)
                plt.subplots_adjust(hspace=0.6)
                plt.axis("off")
                plt.title(f"GroundTruth: {classes[lbs[i]]}\nPredicted: {classes[pred_lbs[i]]}")
                plt.imshow(img)
                if classes[lbs[i]] == classes[pred_lbs[i]]:
                    plt.text(0, 330, "Correct", color="g")
                else:
                    plt.text(0, 330, "Wrong", color="r")

            # plt.savefig("predicted_downloaded.jpg")

    plt.show()


# 預測
# model = fake_vs_real_Model()
# model = models.resnet18()
# model = models.vgg19()
model = models.vgg19_bn()

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    device = "cuda"
    print(f"Using {device}")
    model.to(device)
else:
    device = "cpu"
    print(f"Using {device}")
    model.to(device)


model.eval()
model.load_state_dict(torch.load("Ver_9_VGG19bn_loss_0.13977/fake_vs_real_net.pth"))
# print(model)

outputs = model(images.to(device))

print(outputs)

predict_pair, predicted = torch.max(outputs, 1)

print(torch.max(outputs, 1))

imshow(images, labels, predict=True, pred_lbs=predicted)

# outputs = model(messup_images.to(device))
#
# _, predicted = torch.max(outputs, 1)
#
# imshow(messup_images, messup_labels, predict=True, pred_lbs=predicted, mess_up=True)
