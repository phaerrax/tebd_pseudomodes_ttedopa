#!/usr/bin/julia

using Plots
using DataFrames
using CSV

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")

disablegrifqtech()

isdat(name) = name[end-3:end] == ".dat"

function parseT(name)
  n = replace(name, "t" => "", ".dat" => "")
  return parse(Float64, n)
end

function extract(file, nsites)
  data = DataFrame(CSV.File(file))
  currents = [data[end, Symbol("current_adjsites$n")] for n ∈ 1:nsites+1]
  N = length(currents)
  avg = sum(currents; init=0.0) / N
  stdev = sqrt(sum((currents .- avg).^2; init=0.0) / (N-1))
  return [parseT(basename(file)), avg, stdev]
end

let
  values = mapreduce(permutedims,
                     vcat,
                     [extract(dat, 10)
                      for dat ∈ filter(isdat, readdir(ARGS[1]; join=true))])
  T = values[:,1]
  avg = values[:,2]
  stdev = values[:,3]
  output = Dict(:temperature => T)
  push!(output, :Javg => avg)
  push!(output, :Jstdev => stdev)
  table = DataFrame(output)
  sort!(table, :temperature)
  CSV.write("temperatureplot.dat", table)
  plt = plot(table[!, :temperature], table[!, :Javg])
  savefig(plt, "temperatureplot.png")
end
