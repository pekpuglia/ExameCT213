using FileIO, Plots, ImageShow, Images

function img_to_mat(img::Matrix{RGB{N0f8}})
    #;;; stacks on the 3rd dimension
    return Float32[getproperty.(img, :r);;; getproperty.(img, :g);;; getproperty.(img, :b)]
end

#= HSV convention:
    h::T # Hue in [0,360]
    s::T # Saturation in [0,1]
    v::T # Value in [0,1]
=#

function img_to_mat(mask::Matrix{Gray{N0f8}})
    #each png pixel has either 0x0 or 0x1 
    return Float32.(getproperty.(mask, :val) .> 0)
end

#gentler approach to image Loading: load single batch
function load_batch(batch_size::Int, batch_index::Int, img_width::Int, img_height::Int, path_list::Vector{String})
    #vectorization expects tensor of size (width, height, channels, batch)
    return cat([img_to_mat(imresize(load(path), (img_width, img_height)))
            for path in path_list[(batch_size*(batch_index-1)+1):(batch_index)*batch_size]]..., dims=4)
end