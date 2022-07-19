using Printf, BSON
using Random
using ProgressBars
using FileIO, Plots, ImageShow, Images
using Statistics
using Colors
using CUDA
include("../utils/image.jl")
include("../classification/classification_model.jl")
include("../utils/accuracy.jl")


##
Random.seed!(3)

# Training parameters
batch_size = 16
epochs = 40
train_ratio = 0.85
val_ratio = 1 - train_ratio
num_classes = 2
img_channels = 3
img_width = 254
img_height = 254
img_size = (img_width, img_height)
training_plot = true
dir_fire = "Training/Fire/"
dir_nofire = "Training/No_Fire/"
##

# Creates a vector of all images from their directories
allfiles_fire = [dir_fire*img for img in readdir(dir_fire)]
allfiles_nofire = [dir_nofire*img for img in readdir(dir_nofire)]
labels = vcat(ones(Int8, length(allfiles_fire),1), zeros(Int8, length(allfiles_fire),1))
allfiles_image = vcat(allfiles_fire, allfiles_nofire)


# Partitions the samples in training and validation
total_samples = length(allfiles_image)
@printf "Number of samples: %d\n" total_samples
val_samples = trunc(Int, total_samples*(val_ratio))
#shuffled images and labels still paired
shuffle_perm = randperm(total_samples)
shuffled_images = allfiles_image[shuffle_perm]
shuffled_labels = labels[shuffle_perm]
train_paths = shuffled_images[val_samples+1:end]
val_paths = shuffled_images[1:val_samples]
Ytrain_data = shuffled_labels[val_samples+1:end]
Yval_data = shuffled_labels[1:val_samples]




########################################### 
#    Loading and training the model
###########################################
model = classification_model(img_height, img_width, img_channels)
# Loss function
loss(x,y) = Flux.binarycrossentropy(model(x), y)

# Training options
opt = ADAM(0.01)
last_improvement = 0
losses = []
accs = []
best_acc = 0.0
acc = -1
##

#Xtrain_data = load_classification_images(img_width, img_height, train_paths)
#Xval_data = load_classification_images(img_width, img_height, val_paths)
@info "full dataset loaded"
@info("Beginning training loop...")
for epoch_idx in ProgressBar(1:epochs)
    Xtrain = load_classification_images(batch_size, epoch_idx, img_width, img_height, train_paths)
    Xval = load_classification_images(batch_size, epoch_idx, img_width, img_height, val_paths)
    Ytrain = Ytrain_data[batch_size*epoch_idx+1:(epoch_idx+1)*batch_size]
    Yval = Yval_data[batch_size*epoch_idx+1:(epoch_idx+1)*batch_size]
    # Training for a single epoch
    Flux.train!(loss, Flux.params(model), [(Xtrain, Ytrain)], opt)
    @info "model trained"
    # Ending conditions
    global acc = accuracy(Xval, Yval, model)
    current_loss = loss(Xval,Yval)
    push!(losses, current_loss)
    push!(accs, acc)
    @show acc
    @show current_loss
    if acc >= 0.95
        @info(" -> Early-exiting: We reached 95% accuracy")
        break
    end
    # Saving model if best accuracy
    if  acc >= best_acc
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

#BSON.@load "u_net.bson" params
#Flux.loadparams!(model, params)
plot(1:epochs, losses, title="Loss over epochs", lw=3, label=false)
savefig("loss_2.png")
#@show accuracy(model(val_img_set), val_mask_set, model)