using Printf
using Random
using ProgressBars
using FileIO, Plots, ImageShow
#precisa do TestImages?
using TestImages, Colors, Images
include("../utils/image.jl")
##
Random.seed!(3)

# Training parameters
batch_size = 16
epochs = 30
train_ratio = 0.85
val_ratio = 1 - train_ratio
num_classes = 2
img_channels = 3
img_width = 512
img_height = 512
img_size = (img_width, img_height)
training_plot = true
dir_images = "Images/"
dir_masks = "Masks/"
##

# Creates a vector of all images and masks from their directories
allfiles_image = [img for img in readdir(dir_images)]
allfiles_mask = [img for img in readdir(dir_masks)]

# Partitions the samples in training and validation
total_samples = length(allfiles_image)
@printf "Number of samples: %d\n" total_samples 
val_samples = trunc(Int, total_samples*(val_ratio))
#shuffled masks and images still paired
shuffle_perm = randperm(total_samples)
shuffled_images = allfiles_image[shuffle_perm]
shuffled_masks = allfiles_mask[shuffle_perm]
train_img_paths = shuffled_images[val_samples+1:end]
train_mask_paths = shuffled_masks[val_samples+1:end]
val_img_paths = shuffled_images[1:val_samples]
val_mask_paths = shuffled_masks[1:val_samples]
##
mat = rgb_to_mat(load(dir_images*train_img_paths[1]))
