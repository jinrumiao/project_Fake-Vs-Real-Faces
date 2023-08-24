
# Fake-vs-Real-Faces

- 近來可以在網路上輕易地看到透過AI生成的內容(AI Generated Contents, AIGC)，可能是文章、貼圖或是人像的圖片，那這些內容如果作者沒有特別標註，我們要如何判斷。

- 某天剛好在kaggle上找資料集想要用來試試看自己學習的成效如何，發現了Fake-Vs-Real-Faces (Hard)這一個資料集，就突然有了好像可以用來試試看，用這個資料集訓練出來的成果，是否能解除第一點的疑問。




# 訓練資料集來源及簡介
資料集來源：https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces

![](https://imgur.com/TYmT002.png)

![](https://imgur.com/Ns3e0v1.png)

- 資料集內包含fake與real兩個資料夾，還有一個data.csv檔。
- fake資料夾內有700張300*300由StyleGAN2生成的人臉照片。
- real則有589張同為300*300的真實人臉照片。
- data.csv中為image_ID以及對應的標籤。



# 測試資料集來源

![](https://imgur.com/0DdtH0L.png)

- 001~008: 網路人像照片
- 009~016: 人像修圖前後對比
- 017~020: https://www.nytimes.com/
- 021~024: https://en.photo-ac.com/
- 025~028: Midjourney
- 029~032: StableDiffusion

# 實作流程

![](https://imgur.com/nLbtiRq.png)


# 資料前處理

- 使用Pytorch中的ImageFolder讀取資料套入transforms作Normalize
```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])]
)
# 利用ImageFolder讀取資料，並套用transform
data = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

# 設定train, test, validation要分割的比例後，使用torch.utils.data.random_split分割
train_size, test_size, valid_size = int(len(data) * 0.6), int(len(data) * 0.2), int(len(data) * 0.2)
train_set, test_set, valid_set = random_split(data,
                                              [train_size, test_size + 1, valid_size + 1],
                                              generator=torch.Generator().manual_seed(42))
# 分配到不同的DataLoader中
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True, pin_memory=True)
```

# 模型建立

- 模型1-CNN & FC

```python
class fake_vs_real_Model(nn.Module):
    def __init__(self):
        super(fake_vs_real_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)  # out_size = （300 - 5 + 0）/ 1 + 1  = 296
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 296 / 2 = 148
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=0)  # out_size = （148 - 5 + 0）/ 1 + 1  = 144
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 144 / 2 = 72
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=26, kernel_size=5, padding=0)  # out_size = （72 - 5 + 0）/ 1 + 1  = 68
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 68 / 2 = 34
        self.conv4 = nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, padding=1)  # out_size = （34 - 3 + 2）/ 1 + 1  = 34
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)  # 34 / 2 = 17
        self.fc1 = nn.Linear(32 * 17 * 17, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.output(x)

        return x


model = fake_vs_real_Model()
```

- 模型2-Pytorch resnet18

```python
from torchvision import models

model = models.resnet18()
```

- 模型3-Pytorch vgg19

```python
from torchvision import models

model = models.vgg19()
```
# 模型訓練

- Epoch: 30
- Learning Rate: 0.001
- Optimizer: Adam

```python
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
```
# 測試模型

```python
# 使用訓練好的模型預測test data
model.eval()

test_images, test_labels = next(iter(test_loader))

model.load_state_dict(torch.load(f"loss_{best_loss:.5f}/" + PATH))

outputs = model(test_images.to(device))

_, predicted = torch.max(outputs, 1)

imshow(test_images, test_labels, predict=True, pred_lbs=predicted)
```
# 測試結果

## 模型測試

- 測試模型-1 (training accuracy: 98.8 %、validation accuracy: 98.6 %)
![](https://imgur.com/Cg61LsR.png)
![](https://imgur.com/SYo8vQ4.png)
![](https://imgur.com/UOfe8Sw.png)

- 測試模型-2 (training accuracy: 99.2 %、validation accuracy: 99.4 %)
![](https://imgur.com/geZBdtQ.png)
![](https://imgur.com/kXAbhaq.png)
![](https://imgur.com/RK3rNX5.png)

- 測試模型-3 (training accuracy: 56.4 %、validation accuracy: 52.1 %)
![](https://imgur.com/zKhogxY.png)
![](https://imgur.com/D8IiExj.png)
(因為在程式碼中顯示預測結果方式與dataloader有相關，vgg19顯示卡記憶體使用量較高，batch size無法像前兩個模型一樣設為32，只能設為8所以才只預測8張)
![](https://imgur.com/Fi5xHKw.png)

## 實作結果

![](https://imgur.com/y7BxasG.png)

1、由實驗結果可以看出模型1以及模型2在training、validation和testing的資料及中有滿不錯的結果，但模型3的結果就明顯不如前兩個模型。就以模型2以及模型3兩個總體層數較相近的來比較，可以發現最大的差異是在於模型2使用的resnet18中有加入了Batch Normalization的處理，模型3的vgg19沒有， 而Batch Normalization主要是用來緩解梯度消失或梯度爆炸的問題，所以使用vgg19訓練出來的模型3很有可能就是發生了梯度爆炸的問題。

2、後續有用改寫模型1將卷積層增加到6層後，同樣也會出現梯度爆炸的問題。

3、在torchvision中還有一個模型為vgg19_bn，就是在vgg19的各層中加入Batch Normalization，改用vgg19_bn後預測準確率確實有提升， training accuracy: 0.9806、validation accuracy: 0.9651。
![](https://imgur.com/jeOEWQW.png)
