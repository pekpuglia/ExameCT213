# Accuracy function
function accuracy(x,y,_model)
    acc = 0
    y_hat = Flux.onecold(cpu(y))
    batch_size = size(y_hat, 3)
    for i in 1:batch_size
        acc += sum(Flux.onecold(cpu(_model(x)))[:,:,i] .== Flux.onecold(cpu(y))[:,:,i])/size(y_hat,1)
    end
    return acc = acc/batch_size
end