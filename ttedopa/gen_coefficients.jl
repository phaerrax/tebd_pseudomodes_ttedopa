#!/usr/bin/julia

using DataFrames, CSV
using PseudomodesTTEDOPA

disablegrifqtech()

let
  file = ARGS[1]
  local p
  open(file) do inp
    s = read(inp, String)
    p = JSON.parse(s)
  end
  parameters = merge(p, Dict("filename" => file))

  # Look out 
  fn = parameters["spectral_density_function"]
  tmp = eval(Meta.parse("(a, x) -> " * fn))
  sdf = x -> tmp(parameters["spectral_density_parameters"], x)

  ωc = parameters["frequency_cutoff"]
  n_osc = parameters["number_of_oscillators"]

  # Calcolo dei coefficienti dalla densità spettrale
  T = parameters["temperature"]
  if T == 0
    (Ω, κ, η) = chainmapcoefficients(sdf,
                                     (0, ωc),
                                     n_osc-1;
                                     Nquad=parameters["PolyChaos_nquad"],
                                     discretization=lanczos)
  else
    sdf_thermalised = ω -> thermalisedJ(sdf, ω, T)
    (Ω, κ, η) = chainmapcoefficients(sdf_thermalised,
                                     (-ωc, 0, ωc),
                                     n_osc-1;
                                     Nquad=parameters["PolyChaos_nquad"],
                                     discretization=lanczos)
  end

  dict = Dict()
  push!(dict, :loc => Ω)
  push!(dict, :int => [η; κ])
  table = DataFrame(dict)
  filename = replace(parameters["filename"], ".json" => ".ttedopa.dat")
  CSV.write(filename, table)

  return
end
