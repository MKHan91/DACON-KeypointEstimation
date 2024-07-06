import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import cv2
import os.path as osp
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import argparse

from sklearn.model_selection import KFold
from datetime import datetime
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='Kyepoint Estimation')

parser.add_argument('--root_dir',               type=str,       default=osp.dirname(__file__))

parser.add_argument('--resized_height',         type=int,       default=180)
parser.add_argument('--resized_width',          type=int,       default=320)
parser.add_argument('--num_epochs',             type=int,       default=80)
parser.add_argument('--learning_rate',          type=float,     default=1e-3)

args = parser.parse_args()


def load_csv(data_root):
    df_path = osp.join(data_root, "train_df.csv")
    submission_path = osp.join(data_root, "sample_submission.csv")

    train_df = pd.read_csv(df_path)
    submission = pd.read_csv(submission_path)

    return train_df, submission



def refine_dataset(data_root):
    # 누란된 데이터 제거하는 함수.
    image_paths = glob(osp.join(data_root, 'train_imgs', "*.jpg"))
    
    keypoints_df, submission = load_csv(data_root)
    
    df_image_names = keypoints_df["image"].to_list()
    file_image_names = [path.split('/')[-1] for path in sorted(image_paths)]

    delete_image_names = []
    for file_image in file_image_names:
        if file_image in df_image_names:
            continue
        else:
            delete_image_names.append(file_image)
    
    sampled_image_paths = [img_path for img_path in image_paths if not img_path.split('/')[-1] in delete_image_names]
    
    return sorted(sampled_image_paths), keypoints_df




class KeypointLoader(Dataset):
    def __init__(self, image_paths, keypoints, augmentation, mode='train'):
        self.augmentation = augmentation
        self.mode = mode
        self.image_paths    = image_paths
        self.keypoints_df   = keypoints
    
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_name = image_path.split('/')[-1]
        
        image = Image.open(image_path)
        image = np.array(image)
        resized_image = cv2.resize(image, (args.resized_width, args.resized_height))
        
        if self.augmentation:
            resized_image = self.augmentation(resized_image)
            
        keypoints = self.keypoints_df.iloc[:, 1:49]
        mask = np.array(self.keypoints_df['image'] == image_name)
        keypoints = np.array(keypoints[mask], dtype=np.float32)
        keypoints = np.reshape(keypoints, newshape=(24, 2))
        
        height, width, _ = image.shape
        keypoints[:, 0] = keypoints[:, 0] * (args.resized_height / height)
        keypoints[:, 1] = keypoints[:, 1] * (args.resized_width / width)
        keypoints = np.reshape(keypoints, 48)
        
        return resized_image, keypoints
    
    
    def __len__(self):
        return len(self.image_paths)



class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(96)
        
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(96)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(96 * 22 * 40, 512)  # Adjust the input size based on the input image dimensions
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 48)
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.max_pool(x)
        
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        x = self.max_pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



def main():
    log_dir     = osp.join(args.root_dir, "log")
    model_dir   = osp.join(args.root_dir, "model")
    
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
        
    device = torch.device('cuda')

    # 텐서보드
    writer = SummaryWriter(log_dir)
        
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 데이터 로드
    image_paths, keypoint_df = refine_dataset(data_root=osp.join(args.root_dir, "dataset"))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (indices_train, indices_valid) in enumerate(kf.split(image_paths)):
        if fold != 0: break
        
        model_dir = osp.join(model_dir, f"fold{fold}")
        if not osp.exists(model_dir):
            os.makedirs(model_dir)
        
        # KF cross validation
        train_image_paths   = [image_paths[idx] for idx in indices_train]
        train_keypoints_df  = keypoint_df.loc[indices_train]
        
        valid_image_paths   = [image_paths[idx] for idx in indices_valid]
        valid_keypoints_df  = keypoint_df.loc[indices_valid]
        
        train_dataset = KeypointLoader(image_paths=train_image_paths, keypoints=train_keypoints_df, augmentation=transform, mode='train')
        valid_dataset = KeypointLoader(image_paths=valid_image_paths, keypoints=valid_keypoints_df, augmentation=transform, mode='valid')
        
        train_kpts_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  pin_memory=True)
        valid_kpts_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True)

        # 모델 정의
        model = BaselineCNN()
        model.cuda()
        
        # 최적화
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        
        scaler = torch.cuda.amp.GradScaler()
        
        total_train_steps   = torch.tensor(len(train_kpts_loader), dtype=torch.float32, device=device)
        total_valid_steps   = torch.tensor(len(valid_kpts_loader), dtype=torch.float32, device=device)
        valid_mae_list      = []
        global_step         = 0
        for epoch in range(args.num_epochs):
            
            # ---------------------------- Train ----------------------------
            start = datetime.now()
            
            train_loss = torch.zeros(1,     device=device)
            train_mae  = torch.zeros(1,     device=device)
            model.train()
            for train_step, sample in enumerate(train_kpts_loader):
                optimizer.zero_grad()
                
                inputs = torch.tensor(sample[0], dtype=torch.float32, device=device)
                targets = torch.tensor(sample[1], dtype=torch.float32, device=device)
                
                with torch.cuda.amp.autocast():
                    pred = model(inputs)
                    loss = criterion(pred, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += torch.div(loss, total_train_steps)
                train_mae  += torch.div(torch.sum(torch.mean(torch.abs(pred - targets), axis=0)), total_train_steps)
                
                if train_step % 20 == 0:
                    print(f"Epoch: {'[':>4}{epoch + 1:>4}/{args.num_epochs}] | Step: {train_step+1:>4}/{len(train_kpts_loader)} | MSE Loss: {train_loss.item():>10.4f}")
            
            elapsed_time = datetime.now() - start
            print(f"[TRAIN] Elapsed time: {elapsed_time} | Epoch: [{epoch + 1}/{args.num_epochs}] | MSE Train Loss: {train_loss.item():.4f} | Train MAE: {train_mae.item():.4f}")
            # --------------------------------------------------------------------
            
            
            # ---------------------------- Validation ----------------------------
            val_loss    = torch.zeros(1,   device=device)
            val_mae     = torch.zeros(1,   device=device)
            model.eval()
            for val_sample in valid_kpts_loader:
                
                val_inputs  = torch.tensor(val_sample[0], dtype=torch.float32, device=device)
                val_targets = torch.tensor(val_sample[1], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    val_pred = model(val_inputs)
                    loss = criterion(val_pred, val_targets)
                
                val_loss += torch.div(loss, total_valid_steps)
                val_mae += torch.div(torch.sum(torch.mean(torch.abs(val_pred - val_targets), axis=0)), total_valid_steps)
            
            elapsed_time = datetime.now() - start
            time_left = ((args.num_epochs - epoch+1) * elapsed_time.seconds) / 3600
            
            print(f"[VALID] Time Left: {time_left:15.2f} h | Epoch: [{epoch + 1}/{args.num_epochs}] | MSE Valid Loss: {val_loss.item():.4f} | Valid MAE: {val_mae.item():.4f}")
            # --------------------------------------------------------------------
            
            valid_mae_list.append(val_mae.item())
            if len(valid_mae_list) > 1 and val_mae.item() < min(valid_mae_list[:-1]):
                torch.save(model.state_dict(), osp.join(model_dir, f"baseline_{epoch:03d}.pth"))
                
            
            # 텐서보드 
            writer.add_scalar("Loss/Train Loss", train_loss.item(), global_step=epoch)
            writer.add_scalar("Loss/Valid Loss", val_loss.item(),   global_step=epoch)
            writer.add_scalar("Metric/Train MAE", train_mae.item(), global_step=epoch)
            writer.add_scalar("Metric/Valid MAE", val_mae.item(),   global_step=epoch)


if __name__ == "__main__":
    main()