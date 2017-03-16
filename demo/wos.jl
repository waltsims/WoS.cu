@everywhere function WalkOnSpheres(x0,N)
      d = length(x0)
      h = 1/sqrt(N)
      w = x -> x'*x/(2*d)

      E = 0
      for j = 1:N
          x = x0
          r = Inf
          while r > h
              r = minimum(1-abs.(x))
              s = randn(d)
              x += r*s/norm(s)
          end
          r, m = findmin(1-abs.(x))
          x[m] = sign(x[m])
          E += w(x)
        end
    E/N
  end

"""
    parallel_WalkOnSpheres(x0,N)

Exicutes a parallel implimentation of the Walking on Spheres algorithm

evaluates the Possion Equation with the boundary condition function
x^2/(2*d) at the point x0 by exicuting the Walking on Spheres alrotihm
N times.
"""
  function parallel_WalkOnSpheres(x0,N)
    M = round(Int,N/nworkers())
    E = @parallel (+) for i=1:nworkers()
        WalkOnSpheres(x0,M)
    end
    E/nworkers()
  end

