using Printf, BSON
using Random
using ProgressBars
using FileIO, Plots, ImageShow, Images
using Statistics
using Colors
using CUDA
include("../utils/image.jl")
include("../u_net_translation/u_net_model.jl")
include("../utils/accuracy.jl")


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
########################################### 
#    Loading and training the model
###########################################

model = unet_model(img_height, img_width, img_channels, num_classes)
# Loss function
loss(x,y) = Flux.crossentropy(model(x), y)


# Loading model and dataset onto GPU
# train_img_set = gpu.(train_img_set)
# train_mask_set = gpu.(train_mask_set)
# val_img_set = gpu.(val_img_set)
# val_mask_set = gpu.(val_mask_set)
##

# Training options
last_improvement = 0
losses = []
accs = []
best_acc = 0.0
acc = -1


##
@info("Beginning training loop...")
train_img_set  = load_batch(400, 1,  img_height, img_width, train_img_paths)
train_mask_set = load_batch(400, 1,  img_height, img_width, train_mask_paths)
data = Flux.DataLoader((train_img_set,train_mask_set), batchsize=16)
@info "dataset loaded"

opt = ADAM()
ps = Flux.params(model)
eval_cb() = @show(loss(train_img_set, train_mask_set))
#eval_cb() = @show(loss(train_img_set, train_mask_set))
Flux.@epochs 16 Flux.train!(loss, ps, data, opt, cb = eval_cb)
BSON.@save "u_net.bson" Flux.params(model)
##
###########################################
#           Validating the model
###########################################
val_img_set = load_batch(100, 1, img_height, img_width, val_img_paths)
val_mask_set = load_batch(100, 1, img_height, img_width, val_mask_paths)

BSON.@load "u_net.bson" 
#Flux.loadparams!(model, params)
#plot(1:30, losses, title="Loss over epochs", lw=3, label=false)
#savefig("loss_2.png")
@show accuracy(model(val_img_set), val_mask_set, model)