#!/usr/local/bin/julia
include("wos.jl")

using DataArrays, DataFrames

dimensions =[2, 3, 4, 5, 6, 7,8, 9, 10, 11, 16, 32, 64]#, 128, 256, 257, 532, 1024] 
iterations =[10, 10^2, 10^3, 10^4, 10^5] #,10^6, 10^7, 10^8, 10^9, 10^10] 
results = zeros(length(iterations) * length(dimensions),4)

run = 1
for iteration in iterations
  for dimension in dimensions
	results[run,1] = dimension
	results[run,2] = iteration
	tic()
	results[run,3] = WalkOnSpheres(zeros(dimension), iteration)[1]
	results[run,4] = toq()
	run+=1
  end
end

results = convert(DataFrame, results)
writetable("juliaData.csv",results)




