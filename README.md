# zygote_flux_tutorial

- [GitHub Page](https://atelierarith.github.io/zygote_flux_tutorial/)
- [Release Page](https://github.com/AtelierArith/zygote_flux_tutorial/releases/tag/artifacts%2Flatest)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AtelierArith/zygote_flux_tutorial/HEAD)

# About this repository

- [Julia in Physics 2021 Online](https://akio-tomiya.github.io/julia_in_physics/) に向けて作成した資料置き場.

- 概要: 関数に対しその導関数を数値計算する自動微分は今日の機械学習・深層学習において重要な要素技術になっている. 本チュートリアルでは自動微分をサポートする Zygote.jl を紹介する. さらにその上に構築されている機械学習ライブラリとして Flux.jl を紹介する.


# Usage

## Install Jupyter notebook

```console
$ pip install jupyter jupytext sympy scipy
```

## Install Julia

- Install Julia from [here](https://julialang.org/downloads/)
- Please see [platform specific instructions](https://julialang.org/downloads/platform/) for further installation instructions and if you have trouble installing Julia. 

## Install dependencies for this repository

```julia
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.6.2 (2021-07-14)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```

## Generate slide

```console
$ julia --project=@. -e 'using Remark; Remark.slideshow("slideshow", options = Dict("ratio" => "16:9"), title = "Zygote+Flux+etc")'
```

## View notebook


```console
$ cd /path/to/this/repository
$ jupyter notebook
```
