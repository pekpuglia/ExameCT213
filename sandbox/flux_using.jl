using Flux
##
# learn xor function (lab 6)
# same model architecture
# 2 inputs -> 10 hidden -> 1 output
# backpropagation (alpha = 6.0)
#dataset
target_function(x, y) = (sign(x) == sign(y)) ? true : false
input_xs = (5 * (-1 .+ 2*rand(200)))
input_ys = (5 * (-1 .+ 2*rand(200)))
#formato esperado para vetorização - 1a dimensão é o n de inputs
inputs = [
    input_xs'
    input_ys'
]
outputs = target_function.(input_xs, input_ys)'
positives = inputs[:, outputs']
negatives = inputs[:, .!outputs']
##
model = Chain(
    Dense(2=>10, σ),    #camada pode ser qq função
    Dense(10=>1, σ)
    )
parameters = Flux.params(model)
##
#= cost:
z, a = self.forward_propagation(inputs)
y = expected_outputs
y_hat = a[-1]
cost = np.mean(-(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))
=#
loss(x, y) = Flux.Losses.crossentropy(model(x), y) +
            Flux.Losses.crossentropy(1 .- model(x), .!y)
##
opt = Descent(6.0)
##
#vetor de duplas que serão passadas p loss
data = [(inputs, outputs)]
##
Flux.@epochs 1000 Flux.train!(loss, parameters,
                         data, opt) #- 1 iteração de treinamento
##
using Plots
##
scatter(positives[1,:], positives[2,:], label="Training+", color="red")
scatter!(negatives[1,:], negatives[2,:], label="Training-", color="blue")
##
interval = -5:0.1:5; n = length(interval)
xx = repeat(cat(interval..., dims=2), 1, 1, n)
yy = repeat(cat(interval..., dims=3), 1, n, 1)
z = model([xx;yy])[1,:,:]
contour!(interval, interval, z)


function segmentation_load()

    #=
    This function is not finished. See img_test for the test of this function.
    =#
    batch_size = 16
    epochs = 30
    train_ratio = 0.85
    val_samples = Int()val_ratio = 0.15
    num_classes = 2
    img_channels = 3
    img_width = 512
    img_height = 512
    img_size = (img_width, img_height)
    training_plot = true
    dir_images = "sandbox/Images/"
    dir_masks = "sandbox/Masks/"

    # Creates a vector of all images and masks from their directories
    allfiles_image = [img for img in readdir(dir_images)]
    allfiles_mask = [img for img in readdir(dir_masks)]

    # Partitions the samples in training and validation
    @printf "Number of samples: %d\n" length(allfiles_image)
    total_samples = length(allfiles_image)
    val_samples = trunc(Int, total_samples*(1-train_ratio))

    shuffled_images = shuffle(allfiles_image)
    shuffled_masks = shuffle(allfiles_mask)
    train_img_paths = shuffled_images[val_samples+1:end]
    train_mask_paths = shuffled_masks[val_samples+1:end]
    val_img_paths = shuffled_images[1:val_samples]
    val_mask_paths = shuffled_masks[1:val_samples]

    # This is my attempt to read and parse the images into arrays, but I'm stuck...
    x_train = zeros((length(train_img_paths), img_height, img_width, img_channels))
    y_train = zeros((length(train_mask_paths), img_height, img_width, 1))

    x_val = zeros((length(val_img_paths), img_height, img_width, img_channels))
    y_val = zeros((length(val_mask_paths), img_height, img_width, 1))

    # Loading training images
    image_dir = pwd()*"\\Images\\"
    @printf "Loading training images: %d images ...\n" length(train_img_paths)
    for (n, file_) in ProgressBar(zip(1:length(train_img_paths), train_img_paths))
        file_path = image_dir*file_
        img = load(file_path)
        img = imresize(img, (img_width, img_height))
        mat = channelview(img)
        x_train[n,:,:,:] = permutedims(mat, (2,3,1))
    end

    # Loading training masks
    masks_dir = pwd()*"\\Masks\\"
    @printf "Loading training masks: %d masks ...\n" length(train_mask_paths)
    for (n, file_) in ProgressBar(zip(1:length(train_mask_paths), train_mask_paths))
        file_path = masks_dir*file_
        img = load(file_path)
        img = imresize(img, (img_width, img_height))
        mat = channelview(img)
        y_train[n,:,:,:] = permutedims(mat, (2,3,1))
    end 

    # Loading validation images
    image_dir = pwd()*"\\Images\\"
    @printf "Loading validation images: %d images ...\n" length(val_img_paths)
    for (n, file_) in ProgressBar(zip(1:length(val_img_paths), val_img_paths))
        file_path = image_dir*file_
        img = load(file_path)
        img = imresize(img, (img_width, img_height))
        mat = channelview(img)
        x_val[n,:,:,:] = permutedims(mat, (2,3,1))
    end

    # Loading validation masks
    masks_dir = pwd()*"\\Masks\\"
    @printf "Loading training masks: %d masks ...\n" length(val_mask_paths)
    for (n, file_) in ProgressBar(zip(1:length(val_mask_paths), val_mask_paths))
        file_path = masks_dir*file_
        img = load(file_path)
        img = imresize(img, (img_width, img_height))
        mat = channelview(img)
        y_val[n,:,:,:] = permutedims(mat, (2,3,1))
    end

    # Generate a random image to display
    # TO-DO generate random mask aswell
    random_image = rand(1:length(val_img_paths))
    ImageShow.gif(x_val[random_image,:,:,:])

    ################################################################
    #= 
        TO-DO:
        - Add the model (unet_model(img_height, img_width, img_channels, num_classes))
        - Compile
        - Add model visualization tools 
        - Predict results
        - Plot a few tests (masks vs model)
    =# 
end


function unet_model(img_height, img_width, img_channel, num_classes)
    #=
    This function returns a Flux CNN model for the fire segmentation problem.
        :param img_height: Image height (px)
        :param img_width: Image width (px)
        :param img_channel: Number of channels
        :param num_classes: Number of classes based on the Ground Truth Masks
        :return Flux CNN 
    =#

    #= Conv cheat sheet:
    layer = Conv((Filter_x, Filter_y), input_size => output_size, activation_function)
    example: layer = Conv((5,5), 3 => 7, relu)
    =#

    contracting_block_1 = Chain(
        # First convolutional layer
        Conv((3,3), img_channel => 16, pad=SamePad(), elu),
        Dropout(0.1),
        Conv((3,3), img_channel => 16, pad=SamePad(), elu),
        MaxPool((2,2))
    )

    contracting_block_2 = Chain(
        # Second convolutional layer
        Conv((3,3), 16 => 32, pad=SamePad(), elu),
        Dropout(0.1),
        Conv((3,3), 16 => 32, pad=SamePad(), elu),
        MaxPool((2,2))
    )
    
    contracting_block_3 = Chain(
        # Third convolutional layer
        Conv((3,3), 32 => 64, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 32 => 64, pad=SamePad(), elu),
        MaxPool((2,2))
    )

    contracting_block_4 = Chain(
        # Fourth convolutional layer
        Conv((3,3), 64 => 128, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 64 => 128, pad=SamePad(), elu),
        MaxPool((2,2))
    )

    bottleneck_block = Chain(
        # Bottleneck block
        Conv((3,3), 128 => 256, pad=SamePad(), elu),
        Dropout(0.3),
        Conv((3,3), 128 => 256, pad=SamePad(), elu),
    )

    expanding_block_1 = Chain(
        # First convolutional layer
        ConvTranspose((2,2), 256 => 128, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 128 => 128, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 128 => 128, pad=SamePad(), elu)
    )

    expanding_block_2 = Chain(
        # Second convolutional layer
        ConvTranspose((2,2), 128 => 64, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 64 => 64, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 64 => 64, pad=SamePad(), elu)
    )
    
    expanding_block_3 = Chain(
        # Third convolutional layer
        ConvTranspose((2,2), 64 => 32, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 32 => 32, pad=SamePad(), elu),
        Dropout(0.1),
        Conv((3,3), 32 => 32, pad=SamePad(), elu)
    )

    expanding_block_4 = Chain(
        # Fourth convolutional layer
        ConvTranspose((2,2), 32 => 16, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 16 => 16, pad=SamePad(), elu),
        Dropout(0.1),
        Conv((3,3), 16 => 16, pad=SamePad(), elu)
    )

    output_layer = Chain(
        Conv((1,1), 16 => 1, σ)
    )

    return Chain(
                contracting_block_1,
                    SkipConnection(
                        Chain(contracting_block_2,
                                SkipConnection(
                                    Chain(contracting_block_3,
                                        SkipConnection(
                                            Chain(
                                            contracting_block_4,
                                            bottleneck_block,
                                            expanding_block_1,
                                            expanding_block_2,
                                            expanding_block_3,
                                            expanding_block_4),
                                            (mx, x) -> cat(mx, x, dims=3)
                                        ),
                                        ),
                                    (mx, x) -> cat(mx, x, dims=3))
                                ),
                        (mx, x) -> cat(mx, x, dims=3)
                        ),
                output_layer
                )
end