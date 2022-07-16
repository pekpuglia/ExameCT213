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

