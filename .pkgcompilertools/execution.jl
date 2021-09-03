using Zygote
f(x) = x ^ 3
f'(1)
f'(1.)

f(x,y,z) = x*y*z
gradient(f, 1,2,3)

using Plots

plot(sin)
plot(sin, cos, 0, 2π)
plot(rand(10))
scatter(rand(10))
@animate for i in 1:10
    plot(rand(10))
end
