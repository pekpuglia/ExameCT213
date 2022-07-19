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
        Conv((3,3), img_channel => 16, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.1),
        Conv((3,3), 16 => 16, elu, pad=SamePad(), init = Flux.kaiming_normal),
    )
    cb1_out_c = 16
        
    contracting_block_2 = Chain(       
        MaxPool((2,2)),
        # Second convolutional layer
        Conv((3,3), 16 => 32, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.1),
        Conv((3,3), 32 => 32, elu, pad=SamePad(), init = Flux.kaiming_normal),
    )
    cb2_out_c = 32
    
    contracting_block_3 = Chain(   
        MaxPool((2,2)),
        # Third convolutional layer
        Conv((3,3), 32 => 64, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.2),
        Conv((3,3), 64 => 64, elu, pad=SamePad(), init = Flux.kaiming_normal),
    )
    cb3_out_c = 64
    
    contracting_block_4 = Chain(
        MaxPool((2,2)),
        # Fourth convolutional layer
        Conv((3,3), 64 => 128, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.2),
        Conv((3,3), 128 => 128, elu, pad=SamePad(), init = Flux.kaiming_normal),
    )
    cb4_out_c = 128
    
    bottleneck_block = Chain(
        MaxPool((2,2)),
        # Bottleneck block
        Conv((3,3), 128 => 256, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.3),
        Conv((3,3), 256 => 256, elu, pad=SamePad(), init = Flux.kaiming_normal),
        ConvTranspose((2,2), 256 => 128, elu, stride=(2,2), pad=SamePad())
    )
    btn_out_c = 128

    expanding_block_1 = Chain(
        # First convolutional layer
        Conv((3,3), btn_out_c+cb4_out_c => 128, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.2),
        Conv((3,3), 128 => 128, elu, pad=SamePad(), init = Flux.kaiming_normal),
        ConvTranspose((2,2), 128 => 64, elu, stride=(2,2), pad=SamePad())
    )
    ex1_out_c = 64

    expanding_block_2 = Chain(
        # Second convolutional layer
        Conv((3,3), ex1_out_c+cb3_out_c => 64, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.2),
        Conv((3,3), 64 => 64, elu, pad=SamePad(), init = Flux.kaiming_normal),
        ConvTranspose((2,2), 64 => 32, elu, stride=(2,2), pad=SamePad())
    )
    ex2_out_c = 32
    
    expanding_block_3 = Chain(
        # Third convolutional layer
        # Conv((3,3), ex2_out_c+cb2_out_c => 32, elu, pad=SamePad(), init = Flux.kaiming_normal),
        Conv((3,3), cb2_out_c => 32, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.1),
        Conv((3,3), 32 => 32, elu, pad=SamePad(), init = Flux.kaiming_normal),
        ConvTranspose((2,2), 32 => 16, elu, stride=(2,2), pad=SamePad())
    )
    ex3_out_c = 16

    expanding_block_4 = Chain(
        # Fourth convolutional layer
        Conv((3,3), ex3_out_c+cb1_out_c => 16, elu, pad=SamePad(), init = Flux.kaiming_normal),
        # Dropout(0.1),
        Conv((3,3), 16 => 16, elu, pad=SamePad(), init = Flux.kaiming_normal)
    )
    ex4_out_c = 16

    output_layer = Chain(
        Conv((1,1), ex4_out_c => 1, σ)
    )

    return Chain(
        Chain(contracting_block_1,
            SkipConnection(
                Chain(contracting_block_2,
                    # SkipConnection(
                    #     Chain(contracting_block_3,
                    #         SkipConnection(
                    #             Chain(contracting_block_4,
                    #                 SkipConnection(
                    #                     bottleneck_block,
                    #                     ConcatenateConnection()
                    #                 ),
                    #                 expanding_block_1
                    #             ),
                    #             ConcatenateConnection()
                    #         ),
                    #         expanding_block_2
                    #     ),
                    #     ConcatenateConnection()
                    # ),
                    expanding_block_3
                ),
                ConcatenateConnection()
            ),
            expanding_block_4
        ),
        output_layer
    )
end