.PHONY: slideshow

slideshow:
	julia --project=@. -e 'using Remark; Remark.slideshow("slideshow", options = Dict("ratio" => "16:9"), title = "Zygote+Flux+etc")'

notebook:
	jupytext --to ipynb playground/notebook/julia/*.jl --execute --set-kernel julia-1.9
	jupyter nbconvert --to html playground/notebook/julia/*.ipynb