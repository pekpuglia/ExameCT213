using Flux
#allow for connection output in model_summary
struct ConcatenateConnection
end
(conn::ConcatenateConnection)(mx, x) = cat(mx, x, dims=3)
import Base.string
string(conn::ConcatenateConnection) = "concatenation"

function model_summary(model::Chain, indent = 0)
    for layer in model.layers
        model_summary(layer, indent)
    end
end

function model_summary(model::Conv, indent = 0)
    kernel_x, kernel_y, input_channels, output_channels = size(model.weight)
    println("\t"^indent * "Convolutional layer:")
    println("\t"^indent * "\tKernel size: (" * string(kernel_x) * ", " * string(kernel_y) * ")")
    println("\t"^indent * "\tInput channels: " * string(input_channels))
    println("\t"^indent * "\tOutput channels: " * string(output_channels))
end

function model_summary(model::SkipConnection, indent=0)
    println("\t"^indent * "Skip connection layer:")
    println("\t"^indent * "\tConnection type: " * string(model.connection))
    println("\t"^indent * "\tInternal layer:")
    model_summary(model.layers, indent+1)
end

#caso default (outras camadas)
function model_summary(model, indent=0)
    println("\t"^indent * string(model))
end