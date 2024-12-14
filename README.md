# Image Segmentation

This project demonstrates semantic segmentation using a U-Net model with a ResNet backbone. The application is built using Streamlit for easy interaction and visualization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/longluv1605/Image_segmentation
    cd Image_segmentation
    ```

2. Install the required packages:

    ```sh
    pip install -r streamlit_requirements.txt
    ```

## Usage

To run the Streamlit application, use the following command:

```sh
streamlit run app.py
```

## Model

- UNet
- DS-Net

## Results

- Cityscapes dataset:
  - Unet base: 65% accuracy on validation set
  - Unet with resnet50: 60% accuracy on validation set  
  - Pretrained DSNet: 96% accuracy
- PASCAL-VOC dataset:
  - Unet resnet50: 95% accuracy on train and validation set

## Contributing

- Pham Thanh Long
- Nguyen Duc Minh
- Tran Tien Nam
- Phan Van Hieu
- Vu Minh Hieu
- Nguyen Duc Huy  

## License

IAI-UET license.
