using FileIO, Plots, ImageShow

function rgb_to_mat(img::Matrix{RGB{N0f8}})
    #;;; stacks on the 3rd dimension
    return Float64[getproperty.(img, :r);;; getproperty.(img, :g);;; getproperty.(img, :b)]
end

#= HSV convention:
    h::T # Hue in [0,360]
    s::T # Saturation in [0,1]
    v::T # Value in [0,1]
=#

function mask_to_mat(mask::Matrix{Gray{N0f8}})
    #each png pixel has either 0x0 or 0x1 
    return getproperty.(load(path), :val) .> 0
end

#gentler approach to image Loading: load single batch
function load_batch(batch_size::Int, batch_index::Int, path_list::Vector{String})
    return [img_file_to_mat(path) for path in path_list[batch_size*batch_index+1:(batch_index+1)*batch_size]]
end