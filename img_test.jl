using Printf
using Random
using ProgressBars
using FileIO, Plots, ImageShow
using TestImages, Colors, Images

Random.seed!(3)

# Training parameters
batch_size = 16
epochs = 30
train_ratio = 0.85
val_set_ratio = 0.15
num_classes = 2
img_channels = 3
img_width = 512
img_height = 512
img_size = (img_width, img_height)
training_plot = true
dir_images = "Images/"
dir_masks = "Masks/"


# Creates a vector of all images and masks from their directories
allfiles_image = [img for img in readdir(dir_images)]
allfiles_mask = [img for img in readdir(dir_masks)]

# Partitions the samples in training and validation
@printf "Number of samples: %d\n" length(allfiles_image)
total_samples = length(allfiles_image)
val_samples = trunc(Int, total_samples*(1-train_ratio))

shuffled_images = shuffle(allfiles_image)
shuffled_masks = shuffle(allfiles_mask)
train_img_paths = shuffled_images[1:val_samples]
train_mask_paths = shuffled_masks[1:val_samples]
val_img_paths = shuffled_images[val_samples+1:end]
val_mask_paths = shuffled_masks[val_samples+1:end]

# This is my attempt to read and parse the images into arrays, but I'm stuck...
x_train = zeros((length(train_img_paths), img_height, img_width, img_channels))
y_train = zeros((length(train_mask_paths), img_height, img_width, 1))
image_dir = pwd()*"\\Images\\"
@printf "Loading training images: %d images ...\n" length(train_img_paths)
for (n, file_) in ProgressBar(zip(1:length(train_img_paths), train_img_paths))
    file_path = image_dir*file_
    img = load(file_path)
    mat = channelview(img) # 
    x_train[n] = mat
end