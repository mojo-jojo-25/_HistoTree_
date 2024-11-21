from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF
from datasets.dataset_h5 import  Whole_Slide_Bag_FP
import argparse, os, glob
from PIL import Image
import h5py
import numpy as np
from sklearn.metrics import pairwise_distances
import openslide
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download, HfApi
import configparser


class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        img = img.resize((224, 224))
        sample = {'input': img}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=transform)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def adj_matrix(wsi_coords):
    total = wsi_coords.shape[0]

    patch_distances = pairwise_distances(wsi_coords, metric='euclidean', n_jobs=1)
    neighbor_indices = np.argsort(patch_distances, axis=1)[:, :16]

    adj_s = np.zeros((total, total))

    for i in range(total):
        x_i, y_i = wsi_coords[i][0], wsi_coords[i][1]
        indices = neighbor_indices[i]
        sum = 0
        for j in indices:
            x_j, y_j = wsi_coords[j][0], wsi_coords[j][1]
            if abs(int(x_i) - int(x_j)) <= 2048 and abs(int(y_i) - int(y_j)) <= 2048:
                adj_s[i][j]=1
                sum += 1
            if sum == 9:
                break

    adj_s = torch.from_numpy(adj_s)

    return adj_s

def save_coords(txt_file, coords):
    for coord in coords:
        x, y= coord
        txt_file.writelines(str(x) + '\t' + str(y) + '\n')
    txt_file.close()

def compute_feats( bags_list, model, data_slide_dir, save_path):
    num_bags = len(bags_list)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for i in range(0, num_bags):

        slide_id = os.path.splitext(os.path.basename(bags_list[i]))[0]

        slide_file_path = os.path.join(data_slide_dir, slide_id +'.svs')
        wsi = openslide.open_slide(slide_file_path)
        os.makedirs(os.path.join(save_path,  'simclr_files'), exist_ok=True)

        output_path_file = os.path.join(save_path, 'simclr_files', slide_id)

        # if os.path.exists(output_path_file):
        #     continue

        dataset = Whole_Slide_Bag_FP(file_path=bags_list[i],wsi=wsi, target_patch_size=224, custom_transforms=transform)
        dataloader = DataLoader(dataset=dataset, batch_size=512, collate_fn=collate_features, drop_last=False, shuffle=False)

        wsi_coords=[]
        wsi_feats=[]
        model.eval()
        for count, (batch, coords) in enumerate(dataloader):
            with torch.no_grad():

                batch = batch.to(device, non_blocking=True)
                wsi_coords.append(coords)
                feature_emb = model(batch)

                wsi_feats.extend(feature_emb)


        os.makedirs(os.path.join(save_path, 'simclr_files', slide_id), exist_ok=True)
        wsi_coords = np.vstack(wsi_coords)
        txt_file = open(os.path.join(save_path, 'simclr_files', slide_id, 'c_idx.txt'), "w+")
        save_coords(txt_file, wsi_coords)
        # save node features
        output = torch.stack(wsi_feats, dim=0).cuda()
        print('features size: ', output.shape, flush=True)
        torch.save(output, os.path.join(save_path, 'simclr_files', slide_id, 'features.pt'))
        # save adjacent matrix
        adj_s = adj_matrix(wsi_coords)
        torch.save(adj_s, os.path.join(save_path, 'simclr_files', slide_id, 'adj_s.pt'))
        print('\r Computed: {}/{}'.format(i + 1, num_bags),flush=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def main():
    parser = argparse.ArgumentParser(description='Compute features ')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--dataset', default=None, type=str, help='path to patches')
    parser.add_argument('--weights', default=None, type=str, help='path to the pretrained weights')
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    parser.add_argument('--slide_dir', default=None, type=str, help='path to the output graph folder')
    args = parser.parse_args()


    config = configparser.ConfigParser()
    config.read('configs/config.ini')
    # Ensure to match the section name used in your config.ini
    token = config['HuggingFace']['hf_token']
    api = HfApi()
    user = api.whoami(token)
    print("Logged in as", user['name'])

    local_dir = "uni"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    #hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0)
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

    if torch.cuda.is_available():
        model.cuda()

    os.makedirs(args.output, exist_ok=True)
    bags_list = glob.glob(args.dataset)
    compute_feats(bags_list, model, args.slide_dir, args.output)


if __name__ == '__main__':
    main()
