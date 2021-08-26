.PHONY: slideshow

slideshow:
	julia --project=@. -e 'using Remark; Remark.slideshow("slideshow", options = Dict("ratio" => "16:9"), title = "Zygote+Flux+etc")'