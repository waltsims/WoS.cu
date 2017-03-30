#!/usr/local/bin/julia
using PyPlot
using DataArrays, DataFrames

df = readtable("../docs/data/wos_log.csv")

fig, ax = subplots()
xlabel("number of Paths")
ylabel("execution time (sec)")

for subdf in groupby(df, :nDimensions)
  ax[:loglog](subdf[:nPaths], subdf[:totalTime],marker="o",linewidth=2, alpha=0.6,label=map(string,subdf[1,1]))
end

grid()
ax[:legend](ncol=2,loc="upper left")
show()

fig2, ax2 = subplots()
xlabel("number of paths")
ylabel("relative error")
df2 = df[df[:nDimensions] .== 2,:] 
ax2[:loglog](df2[:nPaths],df2[:relErrror],marker="o",linewidth=2) 
grid()
show()

writetable("dim2Convergence.csv",df[df[:nDimensions] .== 2,[:nDimensions,:nPaths, :relErrror]])

juliadf = readtable("juliaData.csv")
rename!(juliadf, :x1 , :nDimensions) 
rename!(juliadf, :x2, :nPaths)
rename!(juliadf, :x3,:resultCPU)
rename!(juliadf,:x4 , :totalTimeCPU)

scalingdf = join(df, juliadf, on = [:nDimensions, :nPaths], kind = :right)
speedupdf = by(scalingdf, [:nDimensions,:nPaths], df -> df[:totalTimeCPU]./df[:totalTime])
speedupdf = join(scalingdf, speedupdf, on = [:nDimensions, :nPaths],kind = :left)
rename!(speedupdf, :x1, :speedup)

fig3, ax3 = subplots()
for subdf in groupby(speedupdf, :nDimensions)
  ax3[:semilogx](subdf[:nPaths], subdf[:speedup],marker="o",linewidth=2, alpha=0.6,label=map(string,subdf[1,1]))
end
xlabel("number of paths")
ylabel("speedup")

ax3[:legend](ncol=2, loc="upper left")
show()

paths = readtable("../docs/data/paths.csv")
paths = paths[paths[:avgNumSteps] .> 0,:] 
fig4, ax4 = subplots()
for subdf in groupby(paths, :nDimensions)
ax4[:loglog](subdf[:nPaths],subdf[:avgNumSteps],marker="x",linewidth=2, alpha=0.6,label=map(string,subdf[1,1])) 
end 
ax4[:legend](ncol=2,loc="upper left")

ylabel("average number of steps")
xlabel("number of paths")
show()


