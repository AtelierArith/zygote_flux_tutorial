# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Julia 1.6.2
#     language: julia
#     name: julia-1.6
# ---

using Flux
using Flux.Data: DataLoader
using MLDatasets
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# +
xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtest, ytest = MLDatasets.MNIST.testdata(Float32)

xtrain = Flux.unsqueeze(xtrain, 3)
xtest = Flux.unsqueeze(xtest, 3)

ytrain = Flux.onehotbatch(ytrain, 0:9)
ytest = Flux.onehotbatch(ytest, 0:9)

train_loader = DataLoader((xtrain, ytrain), batchsize=128, shuffle=true)
test_loader = DataLoader((xtest, ytest),  batchsize=128)

# +
nclasses = 10
W, H, inC = (28, 28, 1)
out_conv_size = (W ÷ 4 - 3, H ÷ 4 - 3, 16)

model = Chain(
    Conv((5, 5), inC => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    flatten,
    Dense(prod(out_conv_size), 120, relu),
    Dense(120, 84, relu),
    Dense(84, nclasses),
) |> f32
# -

ps = Flux.params(model)
opt = ADAM(0.01)
loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
Flux.@epochs 5 begin
    Flux.train!(loss, ps, train_loader, opt)
    @show(loss(xtest, ytest))
    println("acc", 100 * sum(Flux.onecold(model(xtest)) .== Flux.onecold(ytest)) / size(ytest, 2), "%")
end

println("acc", 100 * sum(Flux.onecold(model(xtest)) .== Flux.onecold(ytest)) / size(ytest, 2), "%")


