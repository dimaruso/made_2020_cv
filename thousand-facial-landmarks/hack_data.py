"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils import data
from torchvision import transforms

from hack_utils import NUM_PTS, CROP_SIZE
from hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from hack_utils import ThousandLandmarksDataset
from hack_utils import restore_landmarks_batch, create_submission

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="mytest")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default="D:\\work\\MADE1\\cv1\\data\\data")
    parser.add_argument("--batch-size", "-b", default=1, type=int)  # 128 is OK for resnet18 finetune @ 6Gb of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float) #default=1e-3
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    i = 0
    good_picture = 0
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):

        """
        get val_loss
        """
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

        """
        add image to new dataset if val_loss < 4
        """
        
        picture = loader.dataset.image_names[i]
        old_image = cv2.imread(picture)
        i += 1
        NEW_PATH0 = "D:\\work\\MADE1\\cv1\\data\\data\\new_train\\images\\"
        OLD_PATH0 = "D:\\work\\MADE1\\cv1\\data\\data\\train\\images\\"
        NEW_PATH = "/home/ubuntu/kaggle_1000facial_points/data/new_train/images/"
        OLD_PATH = "/home/ubuntu/kaggle_1000facial_points/data/train/images/"

        if OLD_PATH in picture:
            picture = NEW_PATH + picture.split(OLD_PATH)[1]
        if loss.item() < 4:
            cv2.imwrite(picture, old_image)
            good_picture += 1


    return good_picture / i

def main(args):
    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
    ])

    print("Reading data...")
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="data")
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                       shuffle=False, drop_last=False)

    print("Creating model...")
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.to(device)
    

    MODEL_FILENAME = "./rexnext300_best.pth"
    with open(MODEL_FILENAME, "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)


    loss_fn = fnn.mse_loss

    # 2. predict for train
    print("Ready for training...")
    print(len(val_dataloader.dataset))

    accuracy = validate(model, val_dataloader, loss_fn, device=device)
    print("good div all: {:5.2}".format(accuracy))



if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))