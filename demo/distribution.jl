#!/usr/local/bin/julia
using PyPlot

#variable initialisation
d = 2;
len = 5000;

uniform = (rand(len,d) - 0.5) * 2;
normal = randn(len, d);

 ucirc = zeros(len,d);
 ncirc = zeros(len,d);
for i=1:len
    ucirc[i,:] = uniform[i,:] ./ norm(uniform[i,:])
end

for i=1:len
    ncirc[i,:] = normal[i,:] ./ norm(normal[i,:])
end

uangles = atan2(ucirc[:,1],ucirc[:,2]);
uangles = map(x -> (x < 0)? abs(x) + pi: abs(x) ,uangles);
nangles = atan2(ncirc[:,1],ncirc[:,2]);
nangles = map(x -> (x < 0)? abs(x) + pi: abs(x) ,nangles);
nbins = 30 # Number of bins


#overwrite distributions for better visability
uniform = (rand(100, d) - 0.5) *2;
normal = randn(100, d);

 #plot histogram comparisons
fig = figure()
fig[:add_subplot](221)
plt[:scatter](normal[:,1],normal[:,2],alpha=0.4, color="blue")
xlabel("x")
ylabel("y")
title("normal distribution")
fig[:add_subplot](222)
plt[:scatter](uniform[:,1],uniform[:,2],alpha=0.4,color="orange")
xlabel("x")
ylabel("y")
title("uniform distribution")
fig[:add_subplot](223)
title("normamlized uniform")
plt[:hist2d](ncirc[:,1],ncirc[:,2],nbins)
xlabel("x")
ylabel("y")
fig[:add_subplot](224)
title(" nomalized normal")
plt[:hist2d](ucirc[:,1],ucirc[:,2],nbins)
xlabel("x")
ylabel("y")
tight_layout()
fig2 = figure()
nbins = 50 # Number of bins
fig2[:add_subplot](121)
title("Angles Normal Distribution")
plt[:hist](nangles, nbins, color="#0f87bf", alpha=0.4, color="blue")
xlabel("Angle (Radians)")
ylabel("Frequency")
fig2[:add_subplot](122)
title("Angle Uniform Distribution")
plt[:hist](uangles,nbins,color="orange", alpha=0.4, color="orange")
xlabel("Angle (Radians)")
ylabel("Frequency")
show()
