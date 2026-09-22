from models.model_zoo import GRIDNET4
from tqdm import tqdm
import os
import numpy as np
import torch
from utils.misc import load_model
from datasets import FPdataset as datasets
from models.model_zoo import GRIDNET4
import torch.backends.cudnn as cudnn
import logging
from torch.utils.data import ConcatDataset
import time


FOLDERS = [
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2000_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2002_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_SPLIT/FVC2004_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2000_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2002_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_UNETENH_SPLIT/FVC2004_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2000_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2002_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB1_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB1_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB2_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB2_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB3_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB3_B",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB4_A",
    "/media/thiago-dias/BACKUP/Datasets/FVC_FLARE_PRIORENH_SPLIT/FVC2004_DB4_B",
    "/media/thiago-dias/BACKUP/Datasets/CrossMatch_FLARE_SPLIT",
    "/media/thiago-dias/BACKUP/Datasets/CrossMatch_FLARE_PRIORENH_SPLIT",
    "/media/thiago-dias/BACKUP/Datasets/CrossMatch_FLARE_UNETENH_SPLIT",
    "/media/thiago-dias/BACKUP/Datasets/UareU_FLARE_SPLIT",
    "/media/thiago-dias/BACKUP/Datasets/UareU_FLARE_PRIORENH_SPLIT",
    "/media/thiago-dias/BACKUP/Datasets/UareU_FLARE_UNETENH_SPLIT",
]
GPU = "0"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_dataloader(dataset_folder):
    logging.info(f"Create the datasets about pose")
    search_folder = os.path.join(dataset_folder, 'image', 'query')
    gallery_folder = os.path.join(dataset_folder, 'image', 'gallery')
    # set the dataset
    e_datasets = []
    for dataset_ in [search_folder, gallery_folder]:
        valid_dataset = datasets.FingerPoseEvalDataset(
            dataset_, 
            img_ppi=500, 
            seg_zoom=4, 
            middle_shape=(512, 512),
            save_folder='VotingPose',
        )
        e_datasets.append(valid_dataset)
    pose_dataset = ConcatDataset(e_datasets)
    pose_dataloader = torch.utils.data.DataLoader(pose_dataset,
                    batch_size=32, shuffle=False,
                    num_workers=16, pin_memory=True)
    return pose_dataloader


def main():
    logging.info(f"Create the model about pose")
    
    model = GRIDNET4(  
        num_pose_2d=(33,33,1),
        num_layers=(64, 128, 256, 512),
        img_ppi=500,
        middle_shape=np.array([512, 512]),
        activate='sigmoid',
        bin_type='invprop',
        with_tv=True,)
    
    model = model.cuda()
    with torch.no_grad():
        model = torch.nn.DataParallel(model) 
    model_path = 'model_weights/VotingPose.pth'
    load_model(model, model_path)

    for dataset_folder in FOLDERS:
        logging.info(f"Processing folder: {dataset_folder}")
        pose_dataloader = build_dataloader(dataset_folder)
        logging.info(f"Start to estimate the pose of the dataset")
        valid_pose(pose_dataloader, model)


def valid_pose(dataloader, model):
    model.eval()
    total_time = 0
    with torch.no_grad():
        with tqdm(total = len(dataloader.dataset), desc=f"EVAl") as pbar:
            for iterx, item in enumerate(dataloader):
                img = item["img"].cuda()
                names = item["name"]
                T_batch = item["T"]
                start = time.time()
                output = model(img)
                total_time += time.time() - start
                B = img.shape[0]
                pbar.update(B)
                for b in range(B):
                    name = names[b]
                    pose_2d = output['pose_2d'][b].detach().cpu().numpy()
                    T = T_batch[b].numpy()
                    T_inv = np.linalg.inv(T)
                    pose_2d[:2] = np.dot(T_inv[:2, :2], pose_2d[:2]) + T_inv[:2, 2]
                    pose_2d[2] = (pose_2d[2] + 180) % 360 - 180 
                    # save the pose_2d
                    mkdir(os.path.dirname(name))
                    np.savetxt(name, pose_2d)
    logging.info(f"Pose estimation finished for {len(dataloader.dataset)} images")
    logging.info(f"Average time for each image is {total_time/len(dataloader.dataset):.2e}s/sample")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    main()