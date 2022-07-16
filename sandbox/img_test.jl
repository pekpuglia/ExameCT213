using Printf
using Random
using ProgressBars
using FileIO, Plots, ImageShow
#precisa do TestImages?
using TestImages, Colors, Images
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
#return displayable array with RGB channels
function rgb_to_mat(img::Matrix{RGB{N0f8}})
    #;;; stacks on the 3rd dimension
    return Float64[getproperty.(img, :r);;; getproperty.(img, :g);;; getproperty.(img, :b)]
end

#= HSV convention:
    h::T # Hue in [0,360]
    s::T # Saturation in [0,1]
    v::T # Value in [0,1]
=#
#=
    todo:
    - train unet model
    - build our model
=#
#return bit matrix
function mask_to_mat(mask::Matrix{Gray{N0f8}})
    #each png pixel has either 0x0 or 0x1 
    return getproperty.(load(path), :val) .> 0
end

#gentler approach to image Loading: load single batch
function load_img_batch(batch_size::Int, batch_index::Int, path_list::Vector{String})
    return [rgb_to_mat(load(path)) for path in path_list[batch_size*batch_index+1:(batch_index+1)*batch_size]]
end
function load_mask_batch(batch_size::Int, batch_index::Int, path_list::Vector{String})
    return [mask_to_mat(load(path)) for path in path_list[batch_size*batch_index+1:(batch_index+1)*batch_size]]
end

##
img,mat = img_file_to_mat(dir_images*train_img_paths[1])
plot(img)