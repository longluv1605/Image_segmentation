import cv2
import torch
import numpy as np
from torch import nn
from PIL import Image
import streamlit as st
from models.unet import Unet_Resnet
from torchvision.transforms import v2
from models.dsnet.models.dsnet import DSNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def mask_to_image(mask, size):
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask_image = Image.fromarray((mask * 255 / mask.max()).astype(np.uint8))
    return mask_image

def predict_unet_res(img, unet_res, device=device):
    #   img_array = np.array(img)
    test_transform = v2.Compose([
        v2.Resize(size=(256, 256), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_transformed = test_transform(img)
    unet_res.eval()
    mask_logits = unet_res(img_transformed.unsqueeze(0).to(device))
    mask = mask_logits.argmax(dim = 1).squeeze().cpu().numpy()
        
    return mask

def predict_dsnet(img, dsnet, device=device):
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_transformed = transform(img)
    dsnet.eval()
    mask_logits = dsnet(img_transformed.unsqueeze(0).to(device))[2][0]
    mask = mask_logits.argmax(0).squeeze().cpu().numpy()
    return mask

def main():        
    unet_res = Unet_Resnet(21).to(device)
    unet_res = nn.DataParallel(unet_res)
    unet_res.load_state_dict(torch.load(
        "save/models/unet-res_epoch30.pth",
        map_location=torch.device('cpu'),
        weights_only = True
    ))
    
    dsnet = DSNet(m=2, n=3, num_classes=19, planes=64, name='m',augment=True).to(device)

    traned_model_state = torch.load("save/models/dsnet_finetuned.pth", map_location=torch.device('cpu'), weights_only=True)
    new_state_dict = {}
    for key, value in traned_model_state.items():
        new_key = key.replace('module.', '')  # Remove 'module.' from the key
        new_state_dict[new_key] = value
    dsnet.load_state_dict(new_state_dict)
    
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
        
        # Đưa vào mô hình
        if split == "pascal":
            mask = predict_unet_res(image, unet_res)
            size = image.size
            mask_image = mask_to_image(mask, size)
        if split == "cityscapes":
            mask = predict_dsnet(image, dsnet)  
            size = image.size
            mask_image = mask_to_image(mask, size)
            
        # Hiển thị ảnh đã tải lên
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption=f"Ảnh gốc {split}", use_container_width=True)

        with col2:
            st.image(mask_image, caption="Ảnh sau xử lý", use_container_width=True)

        st.write("Testing successfully")
        

if __name__ == '__main__':
    main()