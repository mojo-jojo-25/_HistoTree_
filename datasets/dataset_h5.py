from __future__ import print_function, division

import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord


def isBlackPatch(img_np, rgbThresh=20, percentage=0.40):
    gray = np.mean(img_np, axis=-1)  # Convert to grayscale
    black_pixels = np.sum(gray < rgbThresh)  # Count pixels below threshold
    return (black_pixels / gray.size) >= percentage  # Return True if enough pixels are black


def isWhitePatch(img_np, satThresh=200, percentage=0.90):
    gray = np.mean(img_np, axis=-1)
    white_pixels = np.sum(gray > satThresh)
    return (white_pixels / gray.size) >= percentage


def reinhard_normalization(img, ref_means=[146, 127, 132], ref_stds=[26, 14, 19]):
	"""
    Apply Reinhard color normalization to an image.

    Args:
        img: Input image (PIL or NumPy).
        ref_means: Reference means for LAB color space.
        ref_stds: Reference standard deviations for LAB color space.

    Returns:
        Normalized image as NumPy array.
    """
	# Convert image to NumPy array if it's a PIL image
	if not isinstance(img, np.ndarray):
		img = np.array(img)

	# Convert RGB to LAB
	img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

	# Compute mean and standard deviation of the image in LAB space
	img_means, img_stds = cv2.meanStdDev(img_lab)

	# Normalize each channel
	norm_img = np.zeros_like(img_lab, dtype=np.float32)
	for i in range(3):  # LAB has 3 channels
		norm_img[..., i] = ((img_lab[..., i] - img_means[i]) / img_stds[i]) * ref_stds[i] + ref_means[i]

	# Clip values to valid LAB range
	norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)

	# Convert back to RGB
	norm_img = cv2.cvtColor(norm_img, cv2.COLOR_LAB2RGB)

	return norm_img

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
				 file_path,
				 wsi,
				 pretrained=False,
				 custom_transforms=None,
				 custom_downsample=1,
				 target_patch_size=-1
				 ):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained = pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']

			if 'coords' in f:
					dset = f['coords']
					print("Dataset found:", dset)
					print("Available attributes:", list(dset.attrs.keys()))
			else:
					print("Dataset 'coords' not found in the file.")


			self.patch_level = 0
			self.patch_size = 224
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size,) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample,) * 2
			else:
				self.target_patch_size = None
		self.summary()

	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		while True:
			with h5py.File(self.file_path, 'r') as hdf5_file:
				coord = hdf5_file['coords'][idx]
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

			img_np = np.array(img)
			# If the patch is black or white, try the next index
			if isBlackPatch(img_np, rgbThresh=20):
				print ('black')
				idx = (idx + 1) % len(self)  # Move to the next index, wrap around if needed
				continue

			img_np = reinhard_normalization(img_np)

			img = Image.fromarray(img_np)
			if self.target_patch_size is not None:
				img = img.resize(self.target_patch_size)
			img = self.roi_transforms(img).unsqueeze(0)
			return img, coord


class Whole_Slide_Bag_FP_LH(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		w_s = int(256 * (pow(2,  self.patch_level)))
		h_s = int(256 * (pow(2,  self.patch_level)))
		stop_x = coord[0] + w_s
		stop_y = coord[1] + h_s
		high_patches=[]
		for y in range(coord[1], stop_y, 512):
			for x in range(coord[0], stop_x, 512):
				high_patch = self.wsi.read_region((x, y), 1, (256, 256))
				high_patch = self.roi_transforms(high_patch).unsqueeze(0)
				high_patch = high_patch.resize(self.target_patch_size)
				high_patches.append(high_patch)

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord, high_patches

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




