import os

# Select the physical GPU before importing PyTorch.
# If you launch with:
#   CUDA_VISIBLE_DEVICES=1 python test.py ...
# this line is unnecessary and can be removed.

import torch
import torch.nn
import argparse
import math
import utils
import numpy as np
from model.ARWGAN import *
from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import random



def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def yuv_psnr(img):
    imgy = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2:, :, :]
    imgu = -0.14713 * img[:, 0, :, :] + (-0.28886) * img[:, 1, :, :] + 0.436 * img[:, 2:, :, :]
    imgv = 0.615 * img[:, 0, :, :] + -0.51499 * img[:, 1, :, :] + (-0.10001) * img[:, 2:, :, :]
    return imgy, imgu, imgv


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version    : {torch.version.cuda}")
        print(f"Visible GPUs    : {torch.cuda.device_count()}")
        print(f"Current GPU     : {torch.cuda.current_device()}")
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory      : {props.total_memory / (1024**3):.2f} GB")
    else:
        print("Running on CPU.")

    print("=" * 60)


    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source_images', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument("--noise", '-n', nargs="*", action=NoiseArgParser)
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')
    parser.add_argument(
    "--output-dir",
    default="results",
    type=str,
    help="Directory to save evaluation results."
)

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_options, net_config, noise_config = utils.load_options(args.options_file)
    noise_config = args.noise
    noiser = Noiser(noise_config, device)
    checkpoint = torch.load(
    args.checkpoint_file,
    map_location=device,
    weights_only=False
    )

    hidden_net = ARWGAN(net_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    
    source_images = []

    
    for root, _, files in os.walk(args.source_images):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                source_images.append(os.path.join(root,f))

    source_images.sort()
    print(f"Found {len(source_images)} images.")

    results=[]

    for source_image in source_images:
        image_pil = Image.open(source_image).convert("RGB")
        image_pil = image_pil.resize((net_config.H, net_config.W))
        image_tensor = TF.to_tensor(image_pil).to(device)
        image_tensor = image_tensor * 2 - 1
        image_tensor.unsqueeze_(0)
        np.random.seed(42)
        message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                         net_config.message_length))).to(device)

        with torch.inference_mode():
            losses, (encoded_images, noised_images, decoded_messages) = \
                hidden_net.validate_on_batch([image_tensor, message])
        
        

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        message_detached = message.detach().cpu().numpy()
        ber=np.mean(np.abs(decoded_rounded-message_detached))
        accuracy=1-ber

        
        if (len(results) + 1) % 100 == 0:
            print(f"Processed {len(results)} images...")

        results.append({
            "Image":os.path.basename(source_image),
            "BER":ber,
            "Accuracy":accuracy,
            "PSNR":losses["PSNR           "],
            "SSIM":losses["encoded_ssim   "]
        })

    try:

        utils.save_images(
            image_tensor.cpu(),
            encoded_images.cpu(),
            "test",
            output_dir,
            resize_to=(128,128)
        )



    except Exception as e:
        print("save_images skipped:",e)

    print("Finished processing all images.")
    print("Number of results:", len(results))
    
    df = pd.DataFrame(results)

    print("\n==============================")
    print("PER IMAGE RESULTS")
    print("==============================")
    print(df.head())

    print("\n==============================")
    print("OVERALL AVERAGE")
    print("==============================")

    avg = df.mean(numeric_only=True)
    print(avg)


    df.to_csv(
    os.path.join(output_dir, "evaluation_results.csv"),
    index=False
    )


    avg.to_frame().T.to_csv(
    os.path.join(output_dir, "average_metrics.csv"),
    index=False
    )


    print("\nSaved:")
    print(os.path.join(output_dir, "evaluation_results.csv"))
    print(os.path.join(output_dir, "average_metrics.csv"))
    print(os.path.join(output_dir, "epoch-test.png"))



if __name__ == '__main__':
    main()

