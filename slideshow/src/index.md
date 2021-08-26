class: center, middle

# [Julia in Physics 2021 Online](https://akio-tomiya.github.io/julia_in_physics/)

## Zygote.jl/Flux.jl のお話 <span style="font-size: 50%; color: black;"> + 可視化もあるよ</span>

[SatoshiTerasaki](https://github.com/terasakisatoshi)@[AtelierArith](https://sites.google.com/atelier-arith.jp/atelier-arith)

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

# Usage: Univariate function

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

# Usage: Jacobian matrix Part 1

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

# Usage: Jacobian matrix Part 2

ついでに $\iint\exp(-x^2-y^2)dxdy=\pi$ を極座標表示による変数変換後で積分を行うことで確認してみよう.

```julia
julia> using LinearAlgebra, Zygote, HCubature
julia> x(r, θ) = r * cos(θ); y(r, θ) = r * sin(θ)
julia> Φ(r, θ) = [x(r,θ), y(r, θ)]
julia> function J(r, θ)
           jac = jacobian(Φ, r, θ)
           return hcat(jac[1], jac[2])
       end
julia> f(x, y) = exp(-x^2 - y^2)
julia> g(r, θ) = f(x(r, θ), y(r, θ)) * (det(J(r, θ))) # 変数変換 * ヤコビ行列式
julia> g(rθ) = g(rθ[1], rθ[2]) # `hcubature` 関数が受け付けるようにする.
julia> 積分, _ = hcubature(g, [0, 0], [5, 2π]) # r ∈ [0, 5], θ ∈ [0,  2π]
julia> @assert 積分 ≈ π # 右辺は円周率
```

---

# Application: Length of curves

- 半径 $r$ の円周上を動く車の $c = c(t) = (x(t), y(t))=(r \cos t, r \sin t) \in \mathbb{R}^2$ を時刻 $t=0$ からある時刻 $t$ までの移動距離 $s=s(t)$ を求める.

<center>
  <img src=https://user-images.githubusercontent.com/16760547/130846317-48205f82-88c6-4208-bbf4-0a2ad815e63d.gif />
</center>

- 素直に Julia で実装すると次のようになる:

```julia
julia> using Zygote, QuadGK, LinearAlgebra
julia> const r = 2.
julia> p(t) = [r * cos(t), r * sin(t)]
julia> ṗ(t) = jacobian(p, t)[begin] # 戻り値が length=1 の Tuple で来るので中身を取り出す.
julia> s(t) = quadgk(t̃->norm(ṗ(t̃)), 0, t)[begin] # 積分を実行
julia> t = π # \pi + tab で補完
julia> @assert s(t) == r * t
```

- 上記のコードに続いて $s'(t)$ を計算できるとカッコいいところ見せられたが, エラーが生じて動作しない.

---

# Application: Vector fields

```julia
julia> # 前のページの続き
julia> using Plots
julia> t_range = 0:0.5:2π
julia> plot(size=(800,800))
julia> plot!(t->p(t)[1], t->p(t)[2], 0, 2π, aspect_ratio=:equal, legend=false)
julia> quiver!(
  [p(t)[1] for t in t_range], [p(t)[2] for t in t_range], # 始点
  quiver=([ṗ(t)[1] for t in t_range], [ṗ(t)[2] for t in t_range]) # 終点
) 
```

<center>
  <img width="300" alt="Screen Shot 2021-08-26 at 4 06 44" src="https://user-images.githubusercontent.com/16760547/130850395-a550f26d-c904-47a7-9b57-67a9f0530431.png">
</center>

---

# Usage: Hessian matrix part 1

- どうせなので二階微分もしましょう.

```julia
julia> using Zygote
julia> f(x) = 3^x
julia> df(x) = log(3) * 3^x; ddf(x) = log(3)^2 * 3^x
julia> @assert f'(x) ≈ df(x)
julia> @assert f''(x) ≈ ddf(x)
```

- ヘッセ行列 (Hessian matrix) も作れます.

```julia
julia> using Zygote
julia> f(x, y) = sin(x - y)
julia> f(xy) = f(xy[1], xy[2])
julia> x, y = π/2, π/4
julia> h_zygote = hessian(f, [x, y])
julia> h_theoretical = [
          -sin(x-y) sin(x-y) 
          sin(x-y) -sin(x-y)
       ]
julia> @assert h_theoretical ≈ h_zygote
```

---

# Usage: Hessian matrix part 2

高階偏導関数の計算

KdV 方程式 <img src=https://user-images.githubusercontent.com/16760547/130854677-b6eef6d5-97cc-4374-afff-a8ebbf772de9.gif /> の解として
<img src=https://user-images.githubusercontent.com/16760547/130854649-2bbe7a0a-49ea-4061-8ef7-e99ff7529d61.gif width="400"/> なるものが知られている.

```julia
julia> using Zygote
julia> const c = 2
julia> const θ = 6
julia> u(x, t) = (c/2)*(sech(√c / 2 * (x - c * t - θ)))^2
julia> ∂ₓu(x, t) = gradient(u, x, t)[begin] # \partial + tab + \_t + tab
julia> ∂ₜu(x, t) = gradient(u, x, t)[end]
julia> ∂²ₓu(x, t) = gradient(∂ₓu, x, t)[begin] # \partial + tab + \^2 + tab
julia> ∂³ₓu(x, t) = gradient(∂²ₓu, x, t)[begin] # \partial + tab + \^3 + tab
julia> ∂³ₓu(x, t) = hessian(xt -> ∂ₓu(xt[1], xt[2]), [x, t])[1, 1]
julia> ∂ₓu(1., 1.) # 試運転
julia> ∂²ₓu(1., 1.) # ちょっと時間がかかる
julia> ∂³ₓu(1., 1.) # 気長に待つ
julia> x, t = rand(), rand()
julia> @assert abs(∂ₜu(x, t) + 6u(x,t)*∂ₓu(x,t) + ∂³ₓu(x, t)) <  eps(Float64) # 左辺は非常に小さい数になっている.
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

# Zygote.jl に関するここまでのまとめ 

- Julia の中で定義した関数の微分は `using Zygote` を詠唱し適切な関数を呼び出すことで導関数を使うことができてしまった.

---

class: center, middle

# Appendix

---

# Appendix: SymPy.jl

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

# Appendix: with SymEngine.jl

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

---

# Appendix: 高階偏導関数の計算

KdV 方程式 <img src=https://user-images.githubusercontent.com/16760547/130854677-b6eef6d5-97cc-4374-afff-a8ebbf772de9.gif /> の解として
<img src=https://user-images.githubusercontent.com/16760547/130854649-2bbe7a0a-49ea-4061-8ef7-e99ff7529d61.gif width="400"/> なるものが知られている.

```julia
julia> using Zygote
julia> const c = 2
julia> const θ = 6
julia> u(x, t) = (c/2)*(sech(√c / 2 * (x - c * t - θ)))^2
julia> ∂ₓu(x, t) = gradient(u, x, t)[begin] # \partial + tab + \_t + tab
julia> ∂ₜu(x, t) = gradient(u, x, t)[end]
julia> ∂²ₓu(x, t) = gradient(∂ₓu, x, t)[begin] # \partial + tab + \^2 + tab
julia> ∂³ₓu(x, t) = gradient(∂²ₓu, x, t)[begin] # この定義は失敗する
julia> ∂³ₓu(x, t) = hessian(xt -> ∂ₓu(xt[1], xt[2]), [x, t])[1, 1] # こっちにするとうまくいく
julia> ∂ₓu(1., 1.) # 試運転
julia> ∂²ₓu(1., 1.) # ちょっと時間がかかる
julia> ∂³ₓu(1., 1.) # 気長に待つ
julia> x, t = rand(), rand()
julia> @assert abs(∂ₜu(x, t) + 6u(x,t)*∂ₓu(x,t) + ∂³ₓu(x, t)) <  eps(Float64) # 左辺は非常に小さい数になっている.
```