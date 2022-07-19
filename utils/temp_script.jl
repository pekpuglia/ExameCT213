function img_to_mat(mask::Matrix{Gray{N0f8}})
    #each png pixel has either 0x0 or 0x1 
    return getproperty.(mask, :val) .> 0
end


# Loading validation images
function load_val_images(img_width::Int, img_height::Int, path_list::Vector{String})
    return cat([img_to_mat(imresize(load(path), (img_width, img_height)))
            for path in path_list]..., dims=4)
end

val_img_set = load_val_images(img_height, img_width, val_img_paths[1:100])
val_mask_set = load_val_images(img_height, img_width, val_mask_paths[1:100])

BSON.@load "u_net.bson" params
Flux.loadparams!(model, params)
plot(1:30, losses, title="Loss over epochs", lw=3, label=false)
savefig("loss_2.png")
@show accuracy(model(val_img_set), val_mask_set, model)
