include("../utils/model.jl")
################################################################
#= 
    TO-DO:
    - Compile - no need, all models are pure Julia
    - Train
    - Add model visualization tools
    - Predict results
    - Plot a few tests (masks vs model)
=# 
debug_layer(x) =begin
    println(size(x))
    return x
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
        Conv((3,3), 16 => 16, pad=SamePad(), elu),
        MaxPool((2,2))
    )
    cb1_out_c = 16

    contracting_block_2 = Chain(
        
        # Second convolutional layer
        Conv((3,3), 16 => 32, pad=SamePad(), elu),
        Dropout(0.1),
        Conv((3,3), 32 => 32, pad=SamePad(), elu),
        MaxPool((2,2))
    )
    cb2_out_c = 32
    
    contracting_block_3 = Chain(
        
        # Third convolutional layer
        Conv((3,3), 32 => 64, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 64 => 64, pad=SamePad(), elu),
        MaxPool((2,2))
    )
    cb3_out_c = 64

    contracting_block_4 = Chain(
        
        # Fourth convolutional layer
        Conv((3,3), 64 => 128, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 128 => 128, pad=SamePad(), elu),
        MaxPool((2,2))
    )
    cb4_out_c = 128

    bottleneck_block = Chain(
        
        # Bottleneck block
        Conv((3,3), 128 => 256, pad=SamePad(), elu),
        Dropout(0.3),
        Conv((3,3), 256 => 256, pad=SamePad(), elu)
    )
    btn_out_c = 256

    expanding_block_1 = Chain(
        
        # First convolutional layer
        ConvTranspose((2,2), btn_out_c+cb4_out_c => 128, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 128 => 128, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 128 => 128, pad=SamePad(), elu)
    )
    ex1_out_c = 128

    expanding_block_2 = Chain(
        
        # Second convolutional layer
        ConvTranspose((2,2), cb3_out_c+ex1_out_c => 64, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 64 => 64, pad=SamePad(), elu),
        Dropout(0.2),
        Conv((3,3), 64 => 64, pad=SamePad(), elu)
    )
    ex2_out_c = 64
    
    expanding_block_3 = Chain(
        
        # Third convolutional layer
        ConvTranspose((2,2), cb2_out_c+ex2_out_c => 32, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 32 => 32, pad=SamePad(), elu),
        Dropout(0.1),
        Conv((3,3), 32 => 32, pad=SamePad(), elu)
    )
    ex3_out_c = 32

    expanding_block_4 = Chain(
        
        # Fourth convolutional layer
        ConvTranspose((2,2), cb1_out_c+ex3_out_c => 16, stride=(2,2), pad=SamePad(), elu),
        Conv((3,3), 16 => 16, pad=SamePad(), elu),
        Dropout(0.1),
        Conv((3,3), 16 => 16, pad=SamePad(), elu)
    )
    ex4_out_c = 16
    #problema
    output_layer = Chain(
        Conv((1,1), ex4_out_c => 1, σ)
    )

    return Chain(
        Chain(contracting_block_1,
            SkipConnection(
                Chain(contracting_block_2,
                    SkipConnection(
                        Chain(contracting_block_3,
                            SkipConnection(
                                Chain(contracting_block_4,
                                    SkipConnection(
                                        bottleneck_block,
                                        (mx, x) -> cat(mx, x, dims=3)
                                    ),
                                    expanding_block_1
                                ),
                                (mx, x) -> cat(mx, x, dims=3)
                            ),
                            expanding_block_2
                        ),
                        (mx, x) -> cat(mx, x, dims=3)
                    ),
                    expanding_block_3
                ),
                (mx, x) -> cat(mx, x, dims=3)
            ),
            expanding_block_4
        ),
        output_layer
    )
end