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

# # Flux.jl tutorial
#
# - GitHub: https://github.com/FluxML/Flux.jl
# - https://fluxml.ai/Flux.jl/stable/

# ## Load packages

# +
using Random

using Flux
using Plots
# -

] st Flux

# ## Flux.jl で何ができるのか？
#
# ### 自動微分
#
# - [Flux.jl](https://github.com/FluxML/Flux.jl) は [Zygote.jl](https://github.com/FluxML/Zygote.jl) の自動微分をもとに作られている機械学習ライブラリである. Zygote.jl の機能を `using Flux` とすることで利用することができる.
#
# - $f(x) = x^3$ を微分してみよう.

f(x) = x^3
x = 2
@assert f'(x) == adjoint(f)(x) == 3x^2 == 12

# `@which` マクロで調べると `f'` の実行で Zygote.jl の中で定義されている `adjoint` 関数を呼び出していることがわかる.
#
# ```julia
# julia> @which f'
# adjoint(f::Function) in Zygote at ~/.julia/packages/Zygote/nsu1Y/src/compiler/interface.jl:79
# ```
#
# `@less f'` とか `@edit f'` とか試してみると良い

# 実際に動かしてみる
@which f'

# もちろん `gradient` も使える

f(x,y,z) = x*y*z
gradient(f, 1, 2, 3)

# ### 構造体のフィールドに関する微分
#
# - パラメータ付 $w$ を持つ関数 `f=f(x; w)` において, $w$ をうまく調節することで所望の出力を得たいことはよくある. 勾配法によるパラメータの最適化をするとすれば $\frac{\partial f}{\partial w}$ を計算することが必要. Flux.jl ではこういうことができる.
#
# ```julia
# using Zygote
# struct Affine
#            W # weight matrix
#            b # bias vector
#        end
# (layer::Affine)(x) = layer.W * x .+ layer.b
# W = rand(2, 3); b = rand(2); layer = Affine(W, b); x = rand(3)
# gs = gradient(() -> sum(layer(x)), Params([W, b])) # 勾配の計算. do-syntax を使っても良い.
# gs = gradient(Params([W, b])) do
#            sum(layer(x))
#        end
# @assert gs[W] == hcat(x, x)'
# @assert gs[b] == ones(2)
# ```
#
# - もっと言えば TensorFlow や PyTorch などのような深層学習に使われるレイヤー，最適化アルゴリズムを提供している.

# ## ガリレオの実験の Flux 版
#
# - Flux.jl の使い方を示すために [これならわかる機械学習入門](https://www.kspub.co.jp/book/detail/5225493.html) の第１章に出てくるガリレオの実験を拝借する.
#
# ### 問題設定
#
# - 時刻 $t$ での物体の落下距離が $x$ であるデータセット $\{(t_1, x_1),\dots, (t_N, x_N)\}$ が与えられたとする. 例えば $N=8$ として次のように与えられているとする:
#
# |  経過時間(t)  |  落下距離(x) |
# | ---- | ---- |
# |  1  |  1  |
# |  2  |  4  |
# |  3  |  9  |
# |  4  |  16 |
# |  5  |  25 |
# |  6  |  36 |
# |  7  |  49 |
# |  8  |  64 |
#
# 落下距離は落下時間の二次関数になるという事前知識をベースに
# $t$ と $x$ の関係が
#
# $$
# q(t) = q(t; a,b,c) = a t^2 + b t + c
# $$
#
# として得られるだろうという仮説を立てたとする(モデル化). 
#
# - 我々はこのトイモデルにおける係数の $a$, $b$, $c$ を Flux.jl を使って求めたい.
#
# ### 方針
#
# 上記テーブルのデータ $\{(t_1, x_1),\dots, (t_N, x_N)\}$ を用いて平均二乗誤差
#
# $$
# \textrm{loss}(a,b,c) = \frac{1}{N}\sum_{i}^N (q(t_i;a,b,c) - x_i)^2
# $$
#
# を計算する.左辺のパラメータに関する偏微分を計算し下記のように更新する
#
# $$
# \begin{align}
# a \leftarrow a - \eta \frac{\partial \textrm{loss}}{\partial a} \\
# b \leftarrow b - \eta \frac{\partial \textrm{loss}}{\partial b} \\
# c \leftarrow c - \eta \frac{\partial \textrm{loss}}{\partial c}
# \end{align}
# $$
#
# ここで $\eta$ は学習率であり今回のケースでは, 例えば, `0.0008` とする.
#
# この更新規則を繰り返し適用し $\textrm{loss}$ を下げていくことでモデル $q$ のパラメータが所望の結果を出してくれると期待する.
#
# ### 実装に向けて
#
# - 上記の方針は下記のフェーズに分解できる
#   1. モデルの構築: 自前の構造体 or Flux.jl
#   1. 損失関数の定義: 自前の Julia の関数
#   1. 勾配の計算: Flux.jl/Zygote.jl
#   1. 微分を用いて勾配法を実行: Flux.Optimiser

# ### モデルの構築
#
# - 係数がパラメータとして与えれあれている $t$ についての二次関数
#
# $$
# q(t) = q(t; a,b,c) = a t^2 + b t + c
# $$
#
# を表現するために構造体(struct)を定義する. Flux の仕様に合わせるため構造体のフィールドは配列であることを要請することに注意

# +
struct Quadratic
    a
    b
    c
end

# q(x) というように呼び出せるようにしたい
(q::Quadratic)(x) = (q.a * x^2 + q.b * x + q.c)[1]
# -

# - 上記のコードによって次のようなことができる.

# +
a, b, c = rand(), rand(), rand()
q = Quadratic([a], [b], [c])

@assert q.a == [a]
@assert q.b == [b]
@assert q.c == [c]
t = 3
@assert q(t) == a * t^2 + b * t + c
# -

# #### フィールドをパラメータとしてみなす
#
# フィールドの `a`, `b`, `c` をパラメータとみなすためのおまじないをかける必要がある.

Flux.@functor Quadratic

# このようにすることで `Flux.params` を使って Quadratic オブジェクトである `q` が保持するパラメータを列挙することができる.

@show a, b, c
@show ps = Flux.params(q); # Params([[a], [b], [c]]) のように出てくる.

# ### 勾配の計算
#
# 損失関数
#
# $$
# \frac{1}{N}\sum_{i}^N (q(t_i;a,b,c) - x_i)^2
# $$
#
# を定義しパラメータ (ここでは $a, b, c$) に関する微分を計算したい.
#
# `ps` をパラメータとした時
#
# ```julia
# gradient(ps) do
#     # 損失関数の計算
# end
# ```
#
# という Julia のdo syntax を用いて計算する

# +
# データを用意
tdata = [1, 2, 3, 4, 5, 6, 7, 8]
xdata = [1, 4, 9, 16, 25, 36, 49, 64]

# パラメータの初期化
rng = MersenneTwister(1234) # ランダムシードを固定
a, b, c = rand(rng), rand(rng), rand(rng)
q = Quadratic([a], [b], [c])
# `q` からパラメータを取得
ps = Flux.params(q)

# ps の要素に関しての偏微分を計算する.
gs = gradient(ps) do
        # loss function
        x̂ = q.(tdata)
        loss = Flux.mean((x̂ .- xdata) .^ 2)
        loss
     end
# -

# 下記のコードを実行することで `ps` の要素に対しての勾配情報にアクセスできる

for p in ps
    @show gs[p]
end

# ### パラメータ更新をおこなう
#
# 勾配が計算できたのでシンプルな勾配法でパラメータを更新しよう
#
# $$
# \begin{align}
# a \leftarrow a - \eta \frac{\partial \textrm{loss}}{\partial a} \\
# b \leftarrow b - \eta \frac{\partial \textrm{loss}}{\partial b} \\
# c \leftarrow c - \eta \frac{\partial \textrm{loss}}{\partial c}
# \end{align}
# $$
#
# 今回はパラメータが高々 ~~加算~~ ３つしかないので手動で計算もできるが Flux.jl の機能を使う

# 下記のようなコードを実行することでパラメータを更新することができる
#
# ```julia
# ps = Flux.params(yourmodel) # パラメータ取得
# gs = gradient(ps) do # 勾配計算
#          # do something
#          loss = ...
#      end
# η = 0.0008
# opt = Descent(η) # 最適化アルゴリズムを選択
# Flux.Optimise.update!(opt, ps, gs) # ps が更新される
# ```

# +
η = 0.1 # \eta + <tab>
opt = Descent(η) # 学習率が η とするシンプルな勾配法

# update! をする前の情報を保持する
∇loss = deepcopy.([gs[p] for p in ps]) # オブジェクトのコピーとすることが重要
ps_before = deepcopy(ps)
# ps を更新
Flux.Optimise.update!(opt, ps, gs)

# 更新前後の関係を確認
for (δ, p_before, p) in zip(∇loss, ps_before, ps)
    # δ: \delta + <tab>
    @assert p ≈ p_before - η * δ
end
# -

# ### `Flux.Optimise.update!(opt, ps, gs)` がしていること
#
# 実際のFlux.jlのコードを元に単純化/変数名を改変したもので説明する.
#
# `Flux.Optimise.update!(opt, ps, gs)` を実行することで[次のメソッド](https://github.com/FluxML/Flux.jl/blob/v0.12.6/src/optimise/train.jl#L29-L34)が呼ばれる:
#
# ```julia
# function update!(opt, ps::Params, gs)
#   for p in ps
#     update!(opt, p, gs[p]) # ... ★
#   end
# end
# ```
#
# `# ... ★` で示した `update!` は[次のメソッド](https://github.com/FluxML/Flux.jl/blob/v0.12.6/src/optimise/train.jl#L23-L27)である:
#
# ```julia
# function update!(opt, p, Δ)
#   p .-= apply!(opt, p, Δ)
# end
# ```
#
# 上記のコードで `p` を `p .- apply!(opt, p, Δ)` として置き換えていることがわかる.
#
# `apply!` は `opt` の種類(型)によって振る舞いが変わる. ここでは下記のような実装になっている.
#
# ```julia
# mutable struct Descent <: AbstractOptimiser
#   eta::Float64
# end
#
# function apply!(o::Descent, p, Δ)
#   Δ .*= o.eta
# end
# ```
#
# `apply!(opt, p, Δ)` は勾配 `Δ` の `eta` 倍のことである.
#
# 以上をまとめると `p ← p - η * Δ` という更新規則を行なっていることがわかる.
#
# #### Remark
# 厳密なコードは
#
#   - [Flux.jl/src/optimise/train.jl](https://github.com/FluxML/Flux.jl/blob/v0.12.6/src/optimise/train.jl)
#   - [Flux.jl/src/optimise/optimisers.jl](https://github.com/FluxML/Flux.jl/blob/v0.12.6/src/optimise/optimisers.jl)
#   を参照すること

# ## 以上をまとめて実行
#
# - 上記の事柄を一つのセルでまとめて実行する.
# - 学習率は `η = 0.0008` のように十分小さい値を取っている.

# +
# データの用意
tdata = [1,2,3,4,5,6,7,8]
xdata = [1,4,9,16,25,36,49,64]

# セットアップ
rng = MersenneTwister(1234) # ランダムシードを固定
a, b, c = rand(rng), rand(rng), rand(rng)
q = Quadratic([a], [b], [c])
# パラメータを取得
ps = Flux.params(q)
η = 0.0008
# 最適化アルゴリズムを選択
opt = Descent(η)

# 勾配を計算しパラメータ更新をひたすら繰り返す
for iter in 1:1000
    gs = gradient(ps) do
                # loss function
        x̂ = q.(tdata)
        loss = Flux.mean((x̂ .- xdata) .^ 2)
        loss
     end
    Flux.Optimise.update!(opt, ps, gs)
    if mod(iter, 100) == 0
        # loss function
        x̂ = q.(tdata)
        loss = Flux.mean((x̂ .- xdata) .^ 2)
        @show loss
    end
end
# -

# ひとまず `loss` が小さい値に減少していることがわかる. `ps` の値が概ね `[1, 0, 0]` になっていればOK

ps # 概ね [1, 0, 0] に近い値を出すことができているはず.

# ## 結果を確認

p = scatter(tdata, xdata, label="data") # データー
plot!(p, t->q(t), xlim=[0,8], legend=:topleft, label="q") # 予測

# # まとめ
#
# - 簡単なトイモデルによって自前の機械学習モデルを実装することができた.

# # Appendix 
#
# - Flux.jl では MLP/CNN をはじめとする深層学習のモデル構築に必要なレイヤーを提供している
#   - `gradient` や `update!` などの使い方は上記で説明したものと同様である.
#   
# [See flux_mnist](flux_mnist.ipynb)
#
# or 
#
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AtelierArith/zygote_flux_tutorial/HEAD?filepath=playground%2Fnotebook%2Fjulia%2Fflux_mnist.jl)
