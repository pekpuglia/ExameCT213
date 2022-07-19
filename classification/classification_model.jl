include("../utils/model.jl")

debug_layer(x) =begin
    println(size(x))
    return x
end


function classification_model(img_height, img_width, img_channel)
    #=
    This function returns a Flux CNN model for the fire segmentation problem.
        :param img_height: Image height (px)
        :param img_width: Image width (px)
        :param img_channel: Number of channels
        :return Flux CNN 
    =#

    #= Conv cheat sheet:
    layer = Conv((Filter_x, Filter_y), input_size => output_size, activation_function)
    example: layer = Conv((5,5), 3 => 7, relu)
    =#

    first_block = Chain(
        Conv((3,3), img_channel => 8, stride=(2,2)),
        BatchNorm(8, relu)
    )


    second_block = Chain(
        DepthwiseConv((3,3), 8 => 8, stride=(1,2,1,2)),
        BatchNorm(8),
        DepthwiseConv((3,3), 8 => 8, stride=(1,2,1,2)),
        BatchNorm(8, relu),
        MaxPool((3,3), stride=(2,2)),
        debug_layer
    )

    third_block = Chain(
        Conv((3,3), 8 => 8, stride=(2,2)),
        BatchNorm(8, relu)
    )

    skip_block = Chain(Conv((1,1), 8 => 8, stride=(2,2)), debug_layer)

    output_layer = Dense(8, 2, σ)

    return Chain(
        first_block,
        debug_layer,
        Parallel(+; second_block, skip_block),
        debug_layer,
        third_block,
        debug_layer,
        output_layer
    )
end