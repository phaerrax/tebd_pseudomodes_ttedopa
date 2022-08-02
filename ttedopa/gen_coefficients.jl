#!/usr/bin/julia

using Base.Filesystem
using DataFrames
using CSV

# Se lo script viene eseguito su Qtech, devo disabilitare l'output
# grafico altrimenti il programma si schianta.
if gethostname() == "qtech.fisica.unimi.it" ||
  gethostname() == "qtech2.fisica.unimi.it"
  ENV["GKSwstype"] = "100"
  @info "Esecuzione su server remoto. Output grafico disattivato."
else
  delete!(ENV, "GKSwstype")
  # Se la chiave "GKSwstype" non esiste non succede niente.
end

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/tedopa.jl")

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
