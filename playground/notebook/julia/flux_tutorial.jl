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

# +
tdata = [1,2,3,4,5,6,7,8]
xdata = [1,4,9,16,25,36,49,64]

rng = MersenneTwister(1234) # ランダムシードを固定
a, b, c = rand(rng), rand(rng), rand(rng)
q = Quadratic([a], [b], [c])

η = 0.0008
ps = Flux.params(q)
opt = Descent(η)

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

ps # 概ね [1, 0, 0] に近い値を出すことができているはず.

p = scatter(tdata, xdata, label="data") # データー
plot!(p, t->q(t), xlim=[0,8], legend=:topleft, label="q") # 予測
