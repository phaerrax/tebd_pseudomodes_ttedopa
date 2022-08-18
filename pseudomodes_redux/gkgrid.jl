#!/usr/bin/julia

using Plots, DataFrames, CSV, GR
using PseudomodesTTEDOPA

disablegrifqtech()

isdat(name) = name[end-3:end] == ".dat"

function parsegk(name)
  n = replace(name, "g" => "", ".dat" => "")
  v = split(n, "k")
  return parse(Float64, v[1]), parse(Float64, v[2])
end

function extract(file, nsites)
  data = DataFrame(CSV.File(file))
  currents = [data[end, Symbol("current_adjsites$n")] for n ∈ 1:nsites+1]
  N = length(currents)
  avg = sum(currents; init=0.0) / N
  stdev = sqrt(sum((currents .- avg).^2; init=0.0) / (N-1))
  return [parsegk(basename(file))..., avg, stdev]
end

let
  values = mapreduce(permutedims,
                     vcat,
                     [extract(dat, 10)
                      for dat ∈ filter(isdat, readdir(ARGS[1]; join=true))])
  γ = values[:,1]
  κ = values[:,2]
  avg = values[:,3]
  stdev = values[:,4]
  output = Dict(:gamma => γ)
  push!(output, :kappa => κ)
  push!(output, :Javg => avg)
  push!(output, :Jstdev => stdev)
  CSV.write("gkgrid.dat", DataFrame(output))
  N = isqrt(length(γ))
  GR.contourf(γ, κ, avg)
end
