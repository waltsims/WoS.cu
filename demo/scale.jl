#!/usr/local/bin/julia

#change working dir
cd("$(pwd())/..")

#ensure wos is made
run(`make`)
dimensions =[2,4, 8, 16, 32, 64, 128, 256, 528, 1024] 
iterations =[10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9, 10^10] 

for iteration in iterations
  for dimension in dimensions
      run(`./wos -dim "$dimension" -it "$iteration" -p -l`)
  end
end


