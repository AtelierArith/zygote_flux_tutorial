# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
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

# # Zygote Internals
#
# Julia の関数 `f` に対してその微分が `f'` で得られる仕組みとその内部についての簡単な説明をする（簡単だとは言っていない）

# ## Load packages

using Zygote

# ## 復習
#
# 二次関数の微分をすることを考える.

f(x) = x^2

# もちろん, 理論上微分結果は $2x$ となる. さて, これを Julia で実行するには次のように
#
# ```
# f'
# ``` 
#
# とすれば良い.

x = 3
@assert f'(x) == 2x == 6

# ## `f'` の `'`って何？
#
# 実は `'` は `Base.adjoint` 関数の役割を担う.
#
# ドキュメントレベルでは [Punctuation](https://docs.julialang.org/en/v1/base/punctuation/#Punctuation) に説明されている:
#
# > `'`	a trailing apostrophe is the adjoint (that is, the complex transpose) operator Aᴴ
#
# ソースコードレベルでは
#
# Julia本体のリポジトリ [base/operators.jl](https://github.com/JuliaLang/julia/blob/v1.6.2/base/operators.jl#L569) で確認できる:
#
#
# ```julia
# const var"'" = adjoint
# ```
#
# 関数に対する `adjoint(f)` の振る舞いは [Zygote.jl/src/compiler/interface.jl](https://github.com/FluxML/Zygote.jl/blob/v0.6.20/src/compiler/interface.jl#L79) で次のように実装されている:
#
#
# ```julia
# Base.adjoint(f::Function) = x -> gradient(f, x)[1]
# ```

x = 3
adjoint(f)(x) == f'(x)

# ## `gradient` and `pullback`

# `gradient` 関数は大まかに次のようなことをしていると思って良い:
#
# ```julia
# function gradient(f, x)
#   y, back = pullback(f, x)
#   return back(1)
# end
# ```
#
# 厳密には [Zygote.jl/src/compiler/interface.jl](https://github.com/FluxML/Zygote.jl/blob/v0.6.20/src/compiler/interface.jl#L44-L77) を見ること

# `y, back = pullback(f, x)` の戻り値の `y` は `f(x)` のことである. `back` は関数である. これを説明するのに
# 記号を準備する.
#
#

# 変数 $x$ と関数 $f=f(x)$ が与えられているとする. いま, ある関数 $L = L(y)$ where $y=f(x)$ に対して
#
# - $L$ の $x$ に関する微分を $\displaystyle \bar{x} = \frac{dL}{dx}$,
# - $L$ の $y$ に関する微分を $\displaystyle \bar{y} = \frac{dL}{dy}$
#
# と書くことにする. この時, 合成関数の連鎖律から
#
# $$
# \bar{x} = \frac{dL}{dx} = \frac{dL}{dy}\frac{dy}{dx} = \bar{y}\frac{dy}{dx} =: B_x(\bar{y})
# $$
#
# となる. 上述した `back` は $\bar{y}$ を入力とし $\bar{y}\frac{dy}{dx}$ を出力する関数 $B_x(\bar{y})$ になる.
# よって `back(1)` は $\frac{dy}{dx}$ つまり $f'(x)$ を計算していることになる. これは $L(y) = y$ と設定した状況で $\bar{x}$ を計算していることに他ならない.

# - 例えば $f(x)=\sin(x)$ の場合は `gradient` や `pullback` の振る舞いは次のようになっていると思えば良い:
# ただし，ここでは説明を簡単にするために実際の実装と戻り値の型を簡易にしているため厳密には異なる.

# +
square(x) = x * x

pullback_for_sin(x) = (sin(x), ȳ -> ȳ * cos(x))
pullback_for_cos(x) = (cos(x), ȳ -> ȳ * -sin(x))
pullback_for_square(x) = (x * x, ȳ -> ȳ * 2x)

# 自前の pullback 関数
mypullback(::typeof(sin), x) = pullback_for_sin(x) # f=sin の場合の実装
mypullback(::typeof(cos), x) = pullback_for_cos(x) # f=cos の場合の実装
mypullback(::typeof(square), x) = pullback_for_square(x) # f=square の場合の実装

# 自前の gradient 関数
function mygradient(f, x)
    _, back = mypullback(f, x) # f によって振る舞いが異なる.
    return back(1)
end

x = π/6
@assert mygradient(sin, x) ≈ sin'(x) ≈ cos(x)
@assert mygradient(cos, x) ≈ cos'(x) ≈ -sin(x)
@assert mygradient(square, x) ≈ square'(x) ≈ 2x
# -

# Julia が提供しているさまざまな関数に対する `pullback` の実装は `ChainRules.jl`/`ChainRulesCore.jl` にある規則に基づいている.
#
# 三角関数の場合は, 例えば[ここ](https://github.com/JuliaDiff/ChainRules.jl/blob/v1.10.0/src/rulesets/Base/fastmath_able.jl#L12-L16) で見ることができる.
#
# ちなみにタンジェント関数 `tan` に対する微分規則は `@scalar_rule` というマクロを用いて定義している. 
#
# ```julia
# @scalar_rule tan(x) 1 + Ω ^ 2
# ```
#
# これは `Ω = tan(x)` という定義のもとで `1 + Ω²` を微分値として返却する `pullback` を実装するマクロである.
#
# $\tan'(x) = \frac{1}{\cos^2(x)} = 1 + \tan^2(x)$ という数学的性質を利用していることに注意.
#
# Zygote.jl と ChainRules.jl の関係性については 
#
# - JuliaCon2020 の講演 [JuliaCon 2020 | ChainRules.jl | Lyndon White](https://www.youtube.com/watch?v=B4NfkkkJ7rs)
# - JuliaCon2021 の講演 [Everything you need to know about ChainRules 1.0 | Miha Zgubič | JuliaCon2021](https://www.youtube.com/watch?v=a8ol-1l84gc) 
#
# が詳しい
#

x = π/6
tan'(x), 1/(cos(x))^2, 1 + (tan(x))^2

# ## 合成関数の微分
#
# - `back` が関数であることは基本的な関数を合成した関数に対する微分を考える時に便利だからである.
#
# $y=f(x)$, $z=g(y)$ の関係を持つ変数 $x, y, z$ と $L=L(z)$ があるとする. 
#
# この時 $\bar{x}=\frac{dL}{dx}$ は連鎖律によって次のように書ける:
#
# $$
# \begin{aligned}
# \bar{x} &= \frac{dL}{dx} = \frac{dL}{dy}\frac{dy}{dx} = \bar{y} \frac{dy}{dx} = B_x(\bar{y}) \\
# \bar{y} &= \frac{dL}{dy} = \frac{dL}{dz}\frac{dz}{dy} = B_y(\bar{z})
# \end{aligned}
# $$
#
# いま $L=z$ とすれば $\frac{dz}{dx}$ は $(B_x\circ B_y)(\bar{z}=1)$ として計算できる. これが `back` が関数であることの所以である.

# ### Example
#
# $z(x) = \sin(x^2)$ というケースの微分を考える.

# +
square(x) = x * x
f(x) = square(x) # x * x の事.
g(y) = sin(y)
# 合成関数を定義
z(x) = (g ∘ f)(x) # ∘ は \circ とタイプする. もちろん z(x) = g(f(x)) でも良い.

pullback_for_f(x) = (x * x, ȳ -> ȳ * 2x) # (f(x), B_x(ȳ)) みたいなやつ. 中身は pullback_for_square と同じ
pullback_for_g(x) = (sin(x), ȳ -> ȳ * cos(x)) # pullback_for_sin と同じ

mypullback(::typeof(f), x) = pullback_for_f(x)
mypullback(::typeof(g), x) = pullback_for_g(x)

# 合成関数 z に対する pullback の実装
function pullback_for_z(x)
    y, back_for_f = mypullback(f, x)
    z, back_for_g = mypullback(g, y)
    function back_for_z(z̄)
        ȳ = back_for_g(z̄)
        x̄ = back_for_f(ȳ)
        return x̄
    end
    return (z, back_for_z)
end

mypullback(::typeof(z), x) = pullback_for_z(x) # f=z の場合の実装

function mygradient(f, x)
    _, back = mypullback(f, x) # f によって振る舞いが異なる.
    return back(1)
end

x = π/6
@assert mygradient(z, x) ≈ (z)'(x) ≈ 2x * cos(x^2)
# -

# ## Zygote のお仕事
#
# Zygote ではプログラマーが記述したコードを中間表現に変換しその中身を解析することで上記で行なったような `mypullback` 関数を作るお仕事をしている.

# ### `Zygote.@code_ir `
#
# Zygote.jl ではプログラマーが書いた関数を [Static Single Assignment form, SSA](https://ja.wikipedia.org/wiki/%E9%9D%99%E7%9A%84%E5%8D%98%E4%B8%80%E4%BB%A3%E5%85%A5) という中間表現(IR)に変換する.
# ここではプログラマーが記述したプログラムを Julia のコンパイラーにとって優しいものに変換する程度の理解で良い.
#
# 例えば多項式函数は次のように変換される.
#
# ```julia
# julia> poly(x) = 3x*x + 2x + 1
# julia> Zygote.@code_ir poly(1.)
# 1: (%1, %2) # %1 は関数 poly, %2 は引数 x
#   %3 = 3 * %2 # まずは入力を 3倍する i.e. 3x を計算
#   %4 = %3 * %2 # この時点で 3 x^2 を実現
#   %5 = 2 * %2 # 入力を2倍にしたものを計算 i.e. 2x を計算
#   %6 = %4 + %5 + 1 # 諸々を足し合わせる
#   return %6
# ```
#
#
#
# 詳しいことは [IRTools.jl の説明](https://fluxml.ai/IRTools.jl/latest/#Reading-the-IR-1) を見ると良い.
#

poly(x) = 3x*x + 2x + 1
Zygote.@code_ir poly(1.)

# 先ほど定義した `z` についても変換ができる

Zygote.@code_ir z(1.)

# ### Zygote.@code_adjoint
#
# `Zygote.@code_ir` の結果を得て Zygote が生成する `pullback` は `Zygote.@code_adjoint` で確認することができる.
#
# ```julia
# julia> Zygote.@code_adjoint poly(1)
# Zygote.Adjoint(1: (%3, %4 :: Zygote.Context, %1, %2)
#   %5 = Zygote._pullback(%4, Main.:*, 3, %2) 
#   %6 = Base.getindex(%5, 1)
#   %7 = Base.getindex(%5, 2)
#   %8 = Zygote._pullback(%4, Main.:*, %6, %2)
#   %9 = Base.getindex(%8, 1)
#   %10 = Base.getindex(%8, 2)
#   %11 = Zygote._pullback(%4, Main.:*, 2, %2)
#   %12 = Base.getindex(%11, 1)
#   %13 = Base.getindex(%11, 2)
#   %14 = Zygote._pullback(%4, Main.:+, %9, %12, 1)
#   %15 = Base.getindex(%14, 1)
#   %16 = Base.getindex(%14, 2)
#   return %15, 1: (%1)
#   %2 = (@16)(%1)
#   %3 = Zygote.gradindex(%2, 2)
#   %4 = Zygote.gradindex(%2, 3)
#   %5 = (@13)(%4)
#   %6 = Zygote.gradindex(%5, 3)
#   %7 = (@10)(%3)
#   %8 = Zygote.gradindex(%7, 2)
#   %9 = Zygote.gradindex(%7, 3)
#   %10 = (@7)(%8)
#   %11 = Zygote.gradindex(%10, 3)
#   %12 = Zygote.accum(%6, %9, %11)
#   %13 = Zygote.tuple(nothing, %12)
#   return %13)
# ```
#
# - よーく~~（ベニマル）~~目を凝らすと上記で定義した `pullback_for_z` にて定義した表記と似ていることがわかる. 実際,
#
# ```
#   %5 = Zygote._pullback(%4, Main.:*, 3, %2) 
#   %6 = Base.getindex(%5, 1)
#   %7 = Base.getindex(%5, 2)
# ```
#
# は平たくいうと
#
# ```julia
# mul(x, y) = x * y # multiply
# mypullback_for_mul(x, y) = (x * y, Δ -> (Δ *y, Δ * x))
# mypullback(::typeof(mul), x, y) = mypullback_for_mul(x, y)
# ```
#
# という設定のもとで
#
# ```julia
# p5 = mypullback(mul, 3, x)
# p6 = p5[1]
# p5 = p5[2]
# ```
#
# をしていると思えば良い. 配列 `X` に対する添字 `i` へのアクセスは `Base.getindex(X, i)` と書ける.
#
# 上の3 行を 1 行で書くと
#
# ```julia
# out, back = mypullback(mul, 3, x)
# ```
#
# となる. 一方で, 
#
# ```julia
# return %15, 1: (%1)
#   %2 = (@16)(%1)
#   %3 = Zygote.gradindex(%2, 2)
#   %4 = Zygote.gradindex(%2, 3)
#   %5 = (@13)(%4)
#   %6 = Zygote.gradindex(%5, 3)
#   %7 = (@10)(%3)
#   %8 = Zygote.gradindex(%7, 2)
#   %9 = Zygote.gradindex(%7, 3)
#   %10 = (@7)(%8)
#   %11 = Zygote.gradindex(%10, 3)
#   %12 = Zygote.accum(%6, %9, %11)
#   %13 = Zygote.tuple(nothing, %12)
# ```
#
# の部分は `back_for_z` の実装相当のことしていると思えばよい. ここでは, `Zygote.gradindex` は `Bsae.getindex` と同じものという理解で良い.
#
# これで `pullback_for_z` にて定義した表記と似ているということが分かったと思う.
#
# さて, これらから
#
# ```julia
# Zygote.@code_adjoint poly(1)
# ```
#
# の結果から `pullback_for_poly` 関数を定義することが出そうである.

# +
poly(x) = 3x*x + 2x + 1

add(x, y, z) = x + y + z
mul(x, y) = x * y 

pullback_for_add(x, y, z) = (add(x, y, z), Δ -> (Δ, Δ, Δ))
pullback_for_mul(x, y) = (mul(x, y), Δ -> (Δ * y, Δ * x)) # 第二成分は (∂_x mul, ∂_y mul) と思えば良い

mypullback(::typeof(add), x, y, z) = pullback_for_add(x, y, z)
mypullback(::typeof(mul), x, y) = pullback_for_mul(x, y)

function pullback_for_poly(x)
    # %2 は x である

    # %5 = Zygote._pullback(%4, Main.:*, 3, %2)
    # %6 = Base.getindex(%5, 1)
    # %7 = Base.getindex(%5, 2)
    out6, back7 = mypullback(mul, 3, x)
    # %8 = Zygote._pullback(%4, Main.:*, %6, %2)
    # %9 = Base.getindex(%8, 1)
    # %10 = Base.getindex(%8, 2)
    out9, back10 = mypullback(mul, out6, x)
    # %11 = Zygote._pullback(%4, Main.:*, 2, %2)
    # %12 = Base.getindex(%11, 1)
    # %13 = Base.getindex(%11, 2)
    out12, back13 = mypullback(mul, 2, x)
    # %14 = Zygote._pullback(%4, Main.:+, %9, %12, 1)
    # %15 = Base.getindex(%14, 1)
    # %16 = Base.getindex(%14, 2)
    out15, back16 = mypullback(add, out9, out12, 1)
    function back_for_poly(ā)
        # %1 は ā のことである
        # gradindex(∘, 1) は形式的なオブジェクトが入っているので意味のある出力は gradindex(∘,2), gradindex(∘, 3) などである
        # (@13) などは back13 などと対応する
        
        # %2 = (@16)(%1)
        # %3 = Zygote.gradindex(%2, 2)
        # %4 = Zygote.gradindex(%2, 3)
        x̄3, ȳ4 = back16(ā)
        # %5 = (@13)(%4)
        # %6 = Zygote.gradindex(%5, 3)
        _, ȳ6 = back13(ȳ4)
        # %7 = back10(%3)
        # %8 = Zygote.gradindex(%7, 2)
        # %9 = Zygote.gradindex(%7, 3)
        x̄8, ȳ9 = back10(x̄3)
        # %10 = (@7)(%8)
        # %11 = Zygote.gradindex(%10, 3)
        _, ȳ11 = back7(x̄8)
        # %12 = Zygote.accum(%6, %9, %11)
        # %13 = Zygote.tuple(nothing, %12)
        ō12 = sum([ȳ6, ȳ9, ȳ11]) # accum は平たくいうと sum と同じ
        return ō12 # 値はスカラーとして返したいのでここでは tuple に変換はしないでおく
    end
    return out15, back_for_poly
end

mypullback(::typeof(poly), x) = pullback_for_poly(x)

x = rand()
@assert mygradient(poly, x) ≈ (poly)'(x) ≈ 6x + 2
# -

# ### まとめ 
#
# - Zygote.jl の内部でやっていることの説明についてのべた
#
# このノートブックで行っていることがわかれば https://fluxml.ai/Zygote.jl/latest/ のドキュメントも理解できるようになるはずだ.
# 意欲のある方は https://github.com/FluxML/Zygote.jl のソースを追いかけると幸せになれるかもしれない.
