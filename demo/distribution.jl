#!/usr/local/bin/julia
using PyPlot

#variable initialisation
d = 2;
len = 250000;

uniform = (rand(len,d) - 0.5) * 2;
normal = randn(len, d);

ucirc = zeros(len,d);
for i=1:len
    ucirc[i,:] = uniform[i,:] / norm(uniform[i,:])
end

ncirc = zeros(len,d);
for i=1:len
    ncirc[i,:] = normal[i,:] / norm(uniform[i,:])
end

uangles = atan2(ucirc[:,1],ucirc[:,2]);
uangles = map(x -> (x < 0)? abs(x) + pi: abs(x) ,uangles);
nangles = atan2(ncirc[:,1],ncirc[:,2]);
nangles = map(x -> (x < 0)? abs(x) + pi: abs(x) ,nangles);
nbins = 50 # Number of bins


 #plot histogram comparisons
 fig = figure("normal_distribution_histogram",figsize=(10,10))
nh = plt[:hist](nangles, nbins, color="#0f87bf", alpha=0.4)
#ylim(0,100)
xlabel("Angle (Radians)")
ylabel("Frequency")
title("Angle Distribution from Normal Euclidian Distribution")
 fig = figure("uniform_distribution_historgram",figsize=(10,10))
uh = plt[:hist](uangles,nbins,color="orange", alpha=0.4)
#ylim(0,100)
xlabel("Angle (Radians)")
ylabel("Frequency")
title("Angle Distribution from Uniform Euclidian Distribution")
