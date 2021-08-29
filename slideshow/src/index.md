class: center, middle

# [Julia in Physics 2021 Online](https://akio-tomiya.github.io/julia_in_physics/)

## Zygote.jl/Flux.jl のお話 <span style="font-size: 50%; color: black;"> + 可視化もあるよ</span>

[SatoshiTerasaki](https://terasakisatoshi.github.io/)@[AtelierArith](https://sites.google.com/atelier-arith.jp/atelier-arith)

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

class: center, middle

# Zygote.jl のお話

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

## `f'` の `'`って何？

実は `'` は `Base.adjoint` 関数の役割を担う.

ドキュメントレベルでは [Punctuation](https://docs.julialang.org/en/v1/base/punctuation/#Punctuation) に説明されている:

> `'` a trailing apostrophe is the adjoint (that is, the complex transpose) operator Aᴴ

ソースコードレベルでは

Julia本体のリポジトリ [base/operators.jl](https://github.com/JuliaLang/julia/blob/v1.6.2/base/operators.jl#L569) で確認できる:


```julia
const var"'" = adjoint
```

関数に対する `adjoint(f)` の振る舞いは [Zygote.jl/src/compiler/interface.jl](https://github.com/FluxML/Zygote.jl/blob/v0.6.20/src/compiler/interface.jl#L79) で次のように実装されている:


```julia
Base.adjoint(f::Function) = x -> gradient(f, x)[1]
```

---

# Usage: $\nabla f$

- 多変数関数 $f(x, y, z) = xyz$ の勾配 $\nabla f = [f_x, f_y, f_z]^\top$ where $f_x \underset{\textrm{def}}{=} \frac{\partial f}{\partial x}$ etc. を計算したい.
  - もちろん人類は $\nabla f = [yz, zx, xy]^\top$ を知っている.
- Julia では次のように計算する

```julia
julia> using Zygote
julia> f(x, y, z) = x　*　y　*　z # もう後がない。助けてくれ
julia> ∇f(x, y, z) = (y * z, z * x, x * y) # ∇ は \nabla + tab キーで入力できる
julia> x = 3, y = 5, z = 7 # magnum
julia> @assert gradient(f, 3, 5, 7) == ∇f(3, 5, 7) == (35, 21, 15)
```

---

# Application: 

- 正方行列 $X$ に対して行列式 $\det X$ を計算することは $X$ 第 $(i, j)$の成分 $x_{ij}$ 達を変数として見た時の多変数関数とみなせ, 次が成り立つ.

$$
\frac{\partial}{\partial X} \det X 
= \det(X) (X^\top)^{-1}
$$

```julia
julia> using Zygote, SymPy, LinearAlgebra
julia> @vars x11 x12 x13 real=true
julia> @vars x21 x22 x23 real=true
julia> @vars x31 x32 x33 real=true
julia> X = [x11 x12 x13; x21 x22 x23; x31 x32 x33] # SymPy オブジェクトを成分とする行列
3×3 Matrix{Sym}:
 x₁₁  x₁₂  x₁₃
 x₂₁  x₂₂  x₂₃
 x₃₁  x₃₂  x₃₃
julia> gradient(det, X)[begin] # 要素数が 1 の Tuple で返ってるので中身を取り出す.
3×3 Matrix{Sym}:
  x₂₂⋅x₃₃ - x₂₃⋅x₃₂  -x₂₁⋅x₃₃ + x₂₃⋅x₃₁   x₂₁⋅x₃₂ - x₂₂⋅x₃₁
 -x₁₂⋅x₃₃ + x₁₃⋅x₃₂   x₁₁⋅x₃₃ - x₁₃⋅x₃₁  -x₁₁⋅x₃₂ + x₁₂⋅x₃₁
  x₁₂⋅x₂₃ - x₁₃⋅x₂₂  -x₁₁⋅x₂₃ + x₁₃⋅x₂₁   x₁₁⋅x₂₂ - x₁₂⋅x₂₁
julia> @assert gradient(det, X)[begin] == det(X) * inv(X')
julia> X = rand(3, 3); # もちろん入力が数値の場合でもOK
julia> @assert gradient(det, X)[begin] ≈ det(X) * inv(X')
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
2×2 Matrix{Float64}:
 0.866025  -1.0
 0.5        1.73205
julia> J_theoretical = [ 
            cos(θ)  -r * sin(θ)
            sin(θ)  r * cos(θ)
        ]

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
julia> c(t) = [r * cos(t), r * sin(t)]
julia> ċ(t) = jacobian(c, t)[begin] # 戻り値が length=1 の Tuple で来るので中身を取り出す.
julia> s(t) = quadgk(t̃->norm(ċ(t̃)), 0, t)[begin] # 積分を実行
julia> t = π # \pi + tab で補完
julia> @assert s(t) == r * t
```

- 上記のコードに続いて $\dot{s}(t)$ を計算できるとカッコいいが, 残念ながらエラーが生じて動作しない.

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

# Usage: Hessian matrix

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

# Application: 1-Soliton

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

# Appendix: 1-Soliton の可視化

```julia
julia> using Plots
julia> const c = 2; const θ = 6
julia> u(x, t) = (c/2)*(sech(√c / 2 * (x - c * t - θ)))^2
julia> anim = @animate for t in 0:0.1:2
           plot(x->u(x, t), ylim=[0, 1.5], xlim=[2, 15])
       end
julia> gif(anim, "1soliton.gif")
```

<img src=https://user-images.githubusercontent.com/16760547/131127011-0d86a61f-de6a-4fe9-af6e-2cdaddad5592.gif height=300/>

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
julia> W = rand(2, 3); b = rand(2); layer = Affine(W, b); x = rand(3)
julia> gs = gradient(() -> sum(layer(x)), Params([W, b])) # 勾配の計算. do-syntax を使っても良い.
julia> gs = gradient(Params([W, b])) do
           sum(layer(x))
       end
julia> @assert gs[W] == hcat(x, x)'
julia> @assert gs[b] == ones(2)
```

- このような記法により自動微分の機構と Flux.jl をうまく連携できる.

---

# Zygote.jl に関するここまでのまとめ 

- Julia の中で定義した関数の微分は `using Zygote` を詠唱し適切な関数を呼び出すことで導関数を使うことができてしまった.
- 他の Julia パッケージと連携して使う例も紹介した.

## もう少し内部のことを知りたい場合は

- [この資料をご覧ください](zygote_internals.html)

---

class: center, middle

# Flux.jl のお話

---

# Flux.jl

- 自動微分 Zygote.jl の上に構築された機械学習ライブラリ.
- PyTorch のように深層学習のモデル構築に必要な機能を提供
  - MLP/CNN などのレイヤーを提供
  - 最適化アルゴリズム
  - モデルを GPU リソースを用いて学習することもできる
- 自前のモデルを構造体として定義しそれを組み合わせることもできる

- 最近の動向は YouTube で確認できる:
  - [A Tour of the differentiable programming landscape with Flux.jl | Dhairya Gandhi | JuliaCon2021](https://www.youtube.com/watch?v=_UOD4hceFDQ)


---

class: center, middle

# Demo

 [ガリレオ落下実験](flux_tutorial.html)


---

class: center, middle

# Flux.jl のお話

MNIST の学習

---

# Building models

`Chain` というもので基本的なレイヤーを並べてモデルを作ることができる.

```julia
julia> using Flux
julia> nclasses = 10
julia> W, H, inC = (28, 28, 1)
julia> out_conv_size = (W ÷ 4 - 3, H ÷ 4 - 3, 16)
julia> model = Chain(
          Conv((5, 5), inC => 6, relu),
          MaxPool((2, 2)),
          Conv((5, 5), 6 => 16, relu),
          MaxPool((2, 2)),
          flatten,
          Dense(prod(out_conv_size), 120, relu),
          Dense(120, 84, relu),
          Dense(84, nclasses),
    )
julia> model = f32(model) # パラメータの重みを Float32 にする
```

#### Remark

- 多次元配列のデータレイアウトは `(W, H, C)` であることに注意 (インデックスの順番が PyTorch の逆と覚えておけば良い)

---

# DataLoader

MNIST データセットを用意

```julia
julia> using Flux
julia> using Flux.Data: DataLoader
julia> using MLDatasets
julia> xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
julia> xtest, ytest = MLDatasets.MNIST.testdata(Float32)
julia> xtrain = Flux.unsqueeze(xtrain, 3) # (28, 28, 60000) -> (28, 28, 1 , 60000)
julia> xtest = Flux.unsqueeze(xtest, 3)   # (28, 28, 10000) -> (28, 28, 1, 10000)
julia> ytrain = Flux.onehotbatch(ytrain, 0:9) # (60000,) -> (10, 60000)
julia> ytest = Flux.onehotbatch(ytest, 0:9)   # (10000,) -> (10, 10000)
julia> train_loader = DataLoader((xtrain, ytrain), batchsize=128, shuffle=true)
julia> test_loader = DataLoader((xtest, ytest),  batchsize=128)
```

- PyTorch の DataLoader が欲しい場合は [https://github.com/lorenzoh/DataLoaders.jl](https://github.com/lorenzoh/DataLoaders.jl) を見ると良い.


---

# Training models


```julia
julia> ps = Flux.params(model) # モデルのパラメータを取得
jluia> opt = ADAM(0.01) # 最適化アルゴリズムを選択
jluia> loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y) # 損失関数の定義
julia> Flux.@epochs 5 Flux.train!(loss, ps, train_loader, opt) # 5 epoch ぶん回す
jluia> println("acc ", 100 * sum(Flux.onecold(model(xtest)) .== Flux.onecold(ytest)) / size(ytest, 2), "%")
acc98.54%
```

--- 

# Flux.jl まとめ

意外と簡単に作れる.

- [もう少し詳しい資料はこちら](flux_mnist.html)

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

# Appendix: 良い Julia 教材

- [Quantitative Economics with Julia](https://julia.quantecon.org/index_toc.html)
- [JuliaCon YouTube 動画](https://www.youtube.com/results?search_query=Juliacon)
- [Introduction to Computational Thinking](https://computationalthinking.mit.edu/Spring21/)
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html)

---

# Appendix: 困ったら？

- エラーメッセージをキーワードにググる
- Twitter で聞く `#Julia言語`
- [Julia Discourse](https://discourse.julialang.org/) で聞く
- ライブラリのバグであればライブラリを管理するリポジトリの Issue に報告
- Julia 本家の Slack で相談
- 計算機科学そのものを勉強する（リテラシーを身につけるという意味）
- ソフトウェア開発における良いプラクティスを学ぶこと
- 良いコードとは何かについて自分なりの哲学を持つこと
