#!/usr/local/bin/julia
using PyPlot
using DataArrays, DataFrames

df = readtable("../docs/data/wos_log.csv")

fig, ax = subplots()
xlabel("number of Paths")
ylabel("execution time (sec)")

for subdf in groupby(df, :nDimensions)
  ax[:semilogx](subdf[:nPaths], subdf[:totalTime],marker="o", linewidth=2, alpha=0.6,label=map(string,subdf[1,1]))
end

ax[:legend](ncol=2,loc="upper left")
show()

