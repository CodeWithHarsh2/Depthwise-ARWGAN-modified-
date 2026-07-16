# Depthwise-ARWGAN

A parameter-efficient implementation of **ARWGAN (Attention-Guided Robust Image Watermarking based on GAN)** using **Depthwise Separable Convolutions** to reduce computational complexity while maintaining watermarking robustness.

This repository is based on the original **ARWGAN** architecture with modifications to the convolutional blocks for improved parameter efficiency.

---

## Features

- Depthwise Separable Convolutions integrated into Dense Blocks
- Robust image watermark embedding and extraction
- GAN-based encoder-decoder architecture
- Multiple noise attacks for robustness evaluation
- Training checkpoint support
- BER, PSNR and SSIM evaluation
- CSV export of evaluation metrics
- Compatible with GPU and CPU execution

---

## Repository Structure

```
Depthwise-ARWGAN/
│
├── main.py                 # Training launcher
├── train.py                # Training pipeline
├── test.py                 # Model evaluation
├── options.py
├── utils.py
├── noise_argparser.py
├── average_meter.py
├── SSIM.py
├── vgg_loss.py
│
├── model/
│   ├── ARWGAN.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── discriminator.py
│   ├── encoder_decoder.py
│   └── Dense_block.py
│
├── noise_layers/
│
├── checkpoints/
├── logs/
├── results/
├── runs/
└── pretrain/
```

---

## Requirements

Install all required packages

```bash
pip install -r requirements.txt
```

Main dependencies

- Python 3.10+
- PyTorch
- Torchvision
- Kornia (0.5.11)
- NumPy
- OpenCV
- Pillow
- Pandas
- Matplotlib
- TensorBoardX

---

## Dataset Structure

The dataset directory should have the following structure.

```
dataset/
│
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
│
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

The dataset path is supplied using the `--data-dir` argument.

---

## Training

Example

```bash
python main.py new ^
-d "E:\datasets\imagenet-mini" ^
-b 16 ^
-e 100 ^
--name depthwise_arwgan
```

Resume training

```bash
python main.py continue ^
-f runs/depthwise_arwgan_YYYY-MM-DD_HH-MM-SS ^
-e 200
```

---

## Testing

Example

```bash
python test.py ^
-c checkpoints/epoch_100.pth ^
-o runs/<experiment>/options-and-config.pickle ^
-s E:\datasets\imagenet-mini\val ^
-n identity
```

Example with JPEG attack

```bash
python test.py ^
-c checkpoints/epoch_100.pth ^
-o runs/<experiment>/options-and-config.pickle ^
-s E:\datasets\imagenet-mini\val ^
-n jpeg 50
```

---

## Output

### Training

Training automatically saves

```
checkpoints/
```

containing model checkpoints.

Training statistics are stored inside

```
runs/<experiment>/
```

including

- training logs
- validation logs
- configuration
- checkpoints

---

### Evaluation

Testing generates

```
results/
│
├── evaluation_results.csv
├── average_metrics.csv
└── epoch-test.png
```

The reported metrics include

- BER (Bit Error Rate)
- Accuracy
- PSNR
- SSIM

---

## Model Modification

Compared to the original ARWGAN implementation, this repository replaces the standard convolution layers inside the Dense Blocks with **Depthwise Separable Convolutions**, reducing the number of trainable parameters and computational cost while preserving the overall network architecture.

The remaining encoder, decoder, discriminator, training procedure, and loss functions remain consistent with the original implementation.



## Acknowledgements

This work is based on the original ARWGAN implementation.

Original repository:
https://github.com/river-huang/ARWGAN
