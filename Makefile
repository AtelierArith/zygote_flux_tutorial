.PHONY: slideshow

slideshow:
	julia --project=@. -e 'using Remark; Remark.slideshow("slideshow", options = Dict("ratio" => "16:9"), title = "Zygote+Flux+etc")'

notebook:
	jupytext --to ipynb playground/notebook/julia/*.jl --execute
	jupyter nbconvert --to html