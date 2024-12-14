import streamlit as st
from PIL import Image
# import os
# from torchvision import transforms, datasets
import streamlit as st
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import v2
import timm
# import matplotlib.pyplot as plt
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Unet_Resnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.encoder = timm.create_model("resnet50", pretrained=True, features_only=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.block_neck = DoubleConv(2048, 1024)
        self.block_up1 = DoubleConv(1024+1024, 512)
        self.block_up2 = DoubleConv(512+512, 256)
        self.block_up3 = DoubleConv(256+256, 128)
        self.block_up4 = DoubleConv(128+64, 64)
        self.conv_cls = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.block_neck(x5) 
        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.block_up4(x)
        x = self.conv_cls(x) 
        x = self.upsample(x)
        return x
    
model = Unet_Resnet(21).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(
    "unet-res_epoch30.pth",
    map_location=torch.device('cpu'),
    weights_only = True
))

def mask_to_image(mask):
    mask_image = Image.fromarray((mask * 255 / mask.max()).astype(np.uint8))
    return mask_image


def predict(img):
#   img_array = np.array(img)
  test_transform = v2.Compose([
    v2.Resize(size=(256, 256), antialias=True),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  img_transformed = test_transform(img)
  model.eval()
  mask_logits = model(img_transformed.unsqueeze(0).to(device))
  mask = mask_logits.argmax(dim = 1).squeeze().cpu().numpy()

  return mask

# Đặt tiêu đề cho ứng dụng
st.title("Semantic Segmentation Demo")
st.write("Tải lên một ảnh để thử mô hình phân đoạn semantic.")

# Tạo uploader để người dùng tải lên ảnh
uploaded_file = st.file_uploader("Chọn ảnh để phân đoạn", type=["jpg", "png", "jpeg"])

# Kiểm tra nếu người dùng đã tải lên ảnh
if uploaded_file is not None:
    file_name = uploaded_file.name
    split = file_name.split("_")[0]
    image = Image.open(uploaded_file).convert("RGB")
    if split == "pascal":
        mask = predict(image)
        mask_image = mask_to_image(mask)
        
    # Hiển thị ảnh đã tải lên
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption=f"Ảnh gốc {split}", use_container_width=True)

    with col2:
        st.image(mask_image, caption="Ảnh sau xử lý", use_container_width=True)

    st.write("Testing successfully")