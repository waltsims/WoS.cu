#!/usr/local/bin/julia

#change working dir
cd("$(pwd())/..")

#ensure wos is made
run(`make`)
dimensions =[2, 3, 4, 5, 6, 7,8, 9, 10, 11, 16, 32, 64, 128, 256, 257, 532, 1024] 
iterations =[10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9, 10^10] 

for iteration in iterations
  for dimension in dimensions
     #if iteration <= (-dimension *365000000 +1730000000)
      run(`./wos -dim "$dimension" -it "$iteration" -p -l`)
    # end
  end
end


