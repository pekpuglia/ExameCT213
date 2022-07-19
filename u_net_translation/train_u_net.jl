using Printf, BSON
using Random
using ProgressBars
using FileIO, Plots, ImageShow, Images
using Statistics
using Colors
using CUDA
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
########################################### 
#    Loading and training the model
###########################################

model = unet_model(img_height, img_width, img_channels, num_classes)
# Loss function
loss(x,y) = Flux.crossentropy(model(x), y)

# Accuracy function
function accuracy(x,y,_model)
    acc = 0
    y_hat = Flux.onecold(cpu(y))
    batch_size = size(y_hat, 3)
    for i in 1:batch_size
        acc += sum(Flux.onecold(cpu(model(val_img_set)))[:,:,i] .== Flux.onecold(cpu(val_mask_set))[:,:,i])/size(y_hat,1)
    end
    return acc = acc/batch_size
end

# Loading model and dataset onto GPU
# model = gpu(model)
# train_img_set = gpu.(train_img_set)
# train_mask_set = gpu.(train_mask_set)
# val_img_set = gpu.(val_img_set)
# val_mask_set = gpu.(val_mask_set)
##

# Training options
opt = ADAM(0.01)
best_acc = 0.0
last_improvement = 0
parameters = Flux.params(model);
losses = []
accs = []
##
@info("Beginning training loop...")
for epoch_idx in ProgressBar(1:epochs)
    train_img_set  = load_batch(batch_size, epoch_idx,  img_height, img_width, train_img_paths)
    train_mask_set = load_batch(batch_size, epoch_idx,  img_height, img_width, train_mask_paths)
    @info "dataset loaded"
    # Training for a single epoch
    Flux.train!(loss, parameters, [(train_img_set, train_mask_set)], opt)
    @info "trained"
    # Ending conditions
    acc = accuracy(train_img_set, train_mask_set, model)
    current_loss = loss(train_img_set,train_mask_set)
    push!(losses, current_loss)
    push!(accs, acc)
    @show acc
    @show current_loss
    if acc >= 0.95
        @info(" -> Early-exiting: We reached 95% accuracy")
        break
    end
    # Saving model if best accuracy
    if acc >= best_acc
        @info(" -> New best accuracy. Saving model")
        BSON.@save "u_net.bson" params=cpu.(Flux.params(model)) epoch_idx acc
        global best_acc = acc
        global last_improvement = epoch_idx
    end
end
##
###########################################
#           Validating the model
###########################################
BSON.@load "u_net.bson" params
Flux.loadparams!(model, params)
@show accuracy(val_img_set, val_mask_set, model)
lot(1:30, losses, title="Loss over epochs", lw=3, label=false)