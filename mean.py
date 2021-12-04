import  torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
data = datasets.ImageFolder(root = "/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/data/coco/train2017",
                            transform = transforms.ToTensor()
                            )
means = torch.zeros(3)
stds = torch.zeros(3)

for img,labels in data:
    means += torch.mean(img,dim=(1,2))
    stds += torch.std(img,dim=(1,2))

means/=len(data)
stds/=len(data)
print("means:",means)
print("stds:",stds)