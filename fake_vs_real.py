import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import os
from self_models import fake_vs_real_Model

# 資料集路徑
data_path = "./data"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])]
)
# 利用ImageFolder讀取資料，並套用transform
data = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
print(len(data))

# 設定train, test, validation要分割的比例後，使用torch.utils.data.random_split分割
train_size, test_size, valid_size = int(len(data) * 0.6), int(len(data) * 0.2), int(len(data) * 0.2)
train_set, test_set, valid_set = random_split(data,
                                              [train_size, test_size + 1, valid_size + 1],
                                              generator=torch.Generator().manual_seed(42))
# 分配到不同的DataLoader中
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True, pin_memory=True)
# 先取出train_loader中的images跟labels
images, labels = next(iter(train_loader))
# print(images, labels)

classes = ["fake", "real"]
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


# 輸出圖像的函數
# helper function to un-normalize and display an image
def denormalize(image):
    """
    還原Normalize後的照片
    :param image: Normalize後的照片
    :return: 還原後的照片
    """
    image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
    image = image.permute(1, 2, 0)  # Changing from 3x224x224 to 224x224x3
    image = torch.clamp(image, 0, 1)
    return image


def imshow(imgs, lbs, predict=False, pred_lbs=None):
    """
    顯示照片
    :param imgs: 要顯示的照片
    :param lbs: 照片的標籤
    :param predict: 是否是顯示預測後的照片，預設：False
    :param pred_lbs: 當predict=True時，預測出來的類別
    :return:
    """
    if not predict:
        fig = plt.figure(figsize=(20, 10))
        for i, img in enumerate(imgs):
            img = denormalize(img)
            fig.add_subplot(1, 4, i + 1)
            plt.axis("off")
            plt.title(classes[lbs[i]])
            plt.imshow(img)
    else:
        assert (predict == True), "predicted_labels should not be None."
        fig = plt.figure(figsize=(20, 10))
        for i, img in enumerate(imgs):
            img = denormalize(img)
            fig.add_subplot(1, 4, i + 1)
            plt.subplots_adjust(hspace=0.6)
            plt.axis("off")
            plt.title(f"GroundTruth: {classes[lbs[i]]}\nPredicted: {classes[pred_lbs[i]]}")
            plt.imshow(img)
            if classes[lbs[i]] == classes[pred_lbs[i]]:
                plt.text(0, 330, "Correct", color="g")
            else:
                plt.text(0, 330, "Wrong", color="r")

        plt.savefig("predicted.jpg")

    plt.show()


# 顯示圖片
imshow(images, labels)

# 訓練
torch.cuda.empty_cache()
# 模型選用
# model = fake_vs_real_Model()
# model = models.resnet18()
# model = models.vgg19()
model = models.vgg19_bn()
print(model)

# 確認是否有GPU資源
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    device = "cuda"
    print(f"Using {device}")
    model.to(device)
else:
    device = "cpu"
    print(f"Using {device}")
    model.to(device)

LR = 0.001  # learning rate
PATH = 'fake_vs_real_net.pth'
# 紀錄loss & accuracy
train_losses, valid_losses = [], []
train_acc, valid_acc = [], []
best_loss = np.Inf

epochs = 30
for epoch in range(epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loss = 0.0
    valid_loss = 0.0
    train_corrects = 0
    valid_corrects = 0

    # training
    model.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_corrects += torch.sum(preds == labels.data)

        # print statistics
        print(f"epoch: {epoch + 1:4d} → training batch: {i + 1:4d}, loss: {loss.item():.5f}")
        train_loss += loss.item() * inputs.size(0)  # total loss of 1 batch

    # validating
    model.eval()
    for i, data in enumerate(valid_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        valid_corrects += torch.sum(preds == labels.data)

        print(f"epoch: {epoch + 1:4d} → validation batch: {i + 1:4d}, loss: {loss.item():.5f}")
        valid_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    train_accuracy = train_corrects / len(train_loader.dataset)
    train_acc.append(train_accuracy.cpu())

    valid_loss = valid_loss / len(valid_loader.dataset)
    valid_losses.append(valid_loss)

    valid_accuracy = valid_corrects / len(valid_loader.dataset)
    valid_acc.append(valid_accuracy.cpu())
    print(f"epoch: {epoch + 1:4d} Done!!!! → \n"
          f"training loss: {train_loss:.5f}, validating loss: {valid_loss:.5f},\n"
          f"training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}")
    # 模型儲存
    if epoch >= (epochs / 2) and valid_loss < best_loss:
        best_loss = valid_loss
        print(f"saving model.....\npresent best loss: {best_loss}")
        if not os.path.exists(f"loss_{best_loss:.5f}/"):
            os.makedirs(f"loss_{best_loss:.5f}/")
            torch.save(model.state_dict(), f"loss_{best_loss:.5f}/" + PATH)


print('Finished Training')
print(f"Best loss: {best_loss:.5f}")

# 畫出loss & accuracy隨著epoch改變的關係
fig = plt.figure(figsize=(15, 6))
fig.add_subplot(1, 2, 1)
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="train loss")
plt.plot(valid_losses, label="valid loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
fig.add_subplot(1, 2, 2)
plt.title("Training and Validation accuracy")
plt.plot(train_acc, label="train accuracy")
plt.plot(valid_acc, label="valid accuracy")
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("loss & accuracy.jpg")

plt.show()

# 使用訓練好的模型預測test data
model.eval()

test_images, test_labels = next(iter(test_loader))

model.load_state_dict(torch.load(f"loss_{best_loss:.5f}/" + PATH))

outputs = model(test_images.to(device))

_, predicted = torch.max(outputs, 1)

imshow(test_images, test_labels, predict=True, pred_lbs=predicted)
