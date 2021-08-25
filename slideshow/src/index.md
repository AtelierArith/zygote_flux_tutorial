class: center, middle

# [Julia in Physics 2021 Online](https://akio-tomiya.github.io/julia_in_physics/)

## Zygote.jl/Flux.jl のお話 <span style="font-size: 50%; color: black;"> + 可視化もあるよ</span>

SatoshiTerasaki@[AtelierArith](https://sites.google.com/atelier-arith.jp/atelier-arith)

---

# お品書き

## [Zygote.jl](https://github.com/FluxML/Zygote.jl)

- Zygote provides source-to-source automatic differentiation (AD) in Julia, and is the next-gen AD system for the [Flux](https://github.com/FluxML/Flux.jl) differentiable programming framework.

- Julia における__自動微分(=automatic differentiation; AD)__をサポートするパッケージ
- Julia の（__ユーザが定義したものも含む__）関数 $f=f(x)$ に対する導関数 $f'(x)$ を自動で作ってくれる.

## [Flux.jl](https://github.com/FluxML/Flux.jl)

- Flux is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.
- Zygote.jl の自動微分を利用した機械学習, 深層学習のモデルを構築することができる.

---

# Usage: single variable

- 関数 $f(x) = x^2 + x$ に対して導関数 $f'(x)$ を求めたい. 
  - もちろん人類は $f'(x) = 2x + 1$ であることは知っている.
- Julia では次のように計算できる.

```julia
julia> using Zygote # おまじない
julia> f(x) = x^2 + x # ユーザーによって定義された関数
julia> df(x) = 2x + 1 # 理論上こうなってほしい.
julia> f'(3) # 上で定義した f に ' をつけるだけで f に対する導関数を計算する.
julia> @assert f'(3) == df(3) == 7
```

- 関数 $g(x) \underset{\textrm{def}}{=} \exp(f(x))$ に対する微分
  - もちろん人類は $g'(x) = \exp(f(x)) f'(x)$ であることを知っている.

```julia
julia> g(x) = exp(f(x)) # 上の REPL の続き
julia> dg(x) = exp(f(x)) * f'(x) # 理論上こうなる
julia> @assert g'(3) == dg(3) == 1.1392835399330275e6
```

---

# Usage: $\nabla f$

- 多変数関数 $f(x, y, z) = xyz$ の勾配 $\nabla f = [f_x, f_y, f_z]^\top$ where $f_x \underset{\textrm{def}}{=} \frac{\partial f}{\partial x}$ etc. を計算したい.
  - もちろん人類は $\nabla f = [yz, zx, xy]^\top$ を知っている.
- Julia では次のように計算する

```julia
julia> using Zygote
julia> f(x, y, z) = x　*　y　*　z
julia> ∇f(x, y, z) = (y * z, z * x, x * y) # ∇ は \nabla + tab キーで入力できる
julia> # x = 3, y = 4, z = 5 での勾配を計算する
julia> @assert gradient(f, 3, 4, 5) == ∇f(3, 4, 5) == (20, 15, 12)
```

---

# Usage: jacobian matrix

- 曲座標変換 $x=x(r, \theta) = r\cos\theta, y = y(r, \theta)=r\sin\theta$ に対するヤコビ行列を計算したい:

<center>
    <img src=https://user-images.githubusercontent.com/16760547/130817466-cab94837-1d6c-451b-9589-2e2e00f14d64.gif width="20%"/>
</center>

- Julia だと次のようにする:
```julia
julia> using Zygote
julia> x(r, θ) = r * cos(θ); y(r, θ) = r * sin(θ)
julia> f(x, y) = [x, y]; g(r, θ) = f(x(r, θ), y(r, θ))
julia> (r, θ) = (2, π/6)
julia> J_zygote = hcat(jacobian(g, r, θ)...)
julia> J_theoretical = [ 
            cos(θ)  -r * sin(θ)
            sin(θ)  r * cos(θ)
        ]
2×2 Matrix{Float64}:
 0.866025  -1.0
 0.5        1.73205

julia> @assert J_zygote ≈ J_theoretical
```

---

# Usage: Structs and Types

構造体のフィールドオブジェクトを変数と見た時の微分もできる:

<img src=https://user-images.githubusercontent.com/16760547/130832819-981b2c03-2e13-4156-b69d-70447510c5a9.gif height="120"  hspace="100"/>
<img src=https://user-images.githubusercontent.com/16760547/130833654-c4bb6345-4593-4dc6-8dba-2aa0c707e99a.gif height="60"/>

```julia
julia> using Zygote
julia> struct Affine
           W # weight matrix
           b # bias vector
       end
julia> (layer::Affine)(x) = layer.W * x .+ layer.b
julia> W = rand(2, 3); b = rand(2)
julia> layer = Affine(W, b); x = rand(3)
julia> gs = gradient(Params([W, b])) do
           sum(layer(x))
       end
julia> @assert gs[W] == hcat(x, x)'
julia> @assert gs[b] == ones(2)
```

- このような記法により自動微分の機構と Flux.jl とうまく連携できる.

---

class: center, middle

# Appendix

---

# Appendix: SymPy

- Python の sympy の Julia インターフェースを提供する [SymPy.jl](https://github.com/JuliaPy/SymPy.jl) のオブジェクトと連携させることもできる. 
  - 例えば $a$, $b$ がパラメータとし入力データ $x$ に対して得られた $ax + b$ のパラメータに関する偏微分を求めたい.

```julia
julia> using Zygote, SymPy
julia> @vars a b real=true
julia> x = 999
julia> gs = gradient(() -> a * x + b, Params([a, b]))
julia> @assert gs[a] == x == 999
julia> @assert gs[b] == 1
julia> # do-syntax を用いて次のように書くこともできる.
julia> gs = gradient(Params([a, b])) do
           a * x + b
       end
julia> @assert gs[a] == x == 999
julia> @assert gs[b] == 1
```

---

# Appendix: with SymEngine

- SymPy でやったことを [SymEngine.jl](https://github.com/symengine/SymEngine.jl) を使ってでもできる.

```julia
julia> using Zygote, SymEngine
julia> Base.adjoint(x::Basic) = x # おまじない
julia> @vars a b
julia> x = 999
julia> gs = gradient(() -> a * x + b, Params([a, b]))
julia> @assert gs[a] == x == 999
julia> @assert gs[b] == 1
julia> # do-syntax を用いて次のように書くこともできる. そしてよく使う.
julia> gs = gradient(Params([a, b])) do
           a * x + b
       end
julia> @assert gs[a] == x == 999
julia> @assert gs[b] == 1
```
