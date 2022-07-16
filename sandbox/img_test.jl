using Printf, BSON
using Random
using ProgressBars
using FileIO, Plots, ImageShow, Images
using Colors
using Parameters: @with_kw
include("../utils/image.jl")
include("../u_net_translation/u_net_model.jl")


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
allfiles_image = [dir_images*img for img in readdir(dir_images)]
allfiles_mask = [dir_masks*img for img in readdir(dir_masks)]

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

train_img_set = load_batch(batch_size, 1, train_img_paths, "img")
train_mask_set = load_batch(batch_size, 1, train_mask_paths, "mask")
val_img_set = load_batch(batch_size, 1, val_img_paths, "img")
val_mask_set = load_batch(batch_size, 1, val_mask_paths, "mask")
#= train_img_set = rgb_to_mat(load(dir_images*train_img_paths))
train_mask_set = mask_to_mat(load(dir_images*train_img_paths))
val_img_set = rgb_to_mat(load(dir_images*val_img_paths))
val_mask_set = mask_to_mat(load(dir_images*val_img_paths))
 =#
########################################### 
#    Loading and training the model
###########################################

model = unet_model(img_height, img_width, img_channels, num_classes)
# Loss function
loss(x,y) = Flux.logitcrossentropy(model(x), y)

# Loading model and dataset onto GPU
model = gpu(model)
train_img_set = gpu.(train_img_set)
train_mask_set = gpu.(train_mask_set)
val_img_set = gpu.(val_img_set)
val_mask_set = gpu.(val_mask_set)


# Training options
opt = ADAM(0.01)
@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in ProgressBar(1:epochs)
    # Training for a single epoch
    Flux.train!(loss, Flux.params(model), [(train_img_set, train_mask_set)], opt)
    # Terminate if NaN
    if anynan(paramvec(model))
        @error "NaN params"
        break
    end
    # Ending conditions
    if acc >= 0.95
        @info(" -> Early-exiting: We reached 95% accuracy")
        break
    end
    # Saving model if best accuracy
    if acc >= best_acc
        @info(" -> New best accuracy. Saving model")
        BSON.@save joinpath(args.savepath, "u_net.bson") params=cpu.(params(model)) epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end
end

###########################################
#           Validating the model
###########################################
BSON.@load joinpath(args.savepath, "u_net.bson") params
Flux.loadparams!(model, params)
accuracy(x,y,model) = mean(onecold(cpu(model(x)))) .== onecold(cpu(y))
@show accuracy(val_img_set, val_mask_set, model)