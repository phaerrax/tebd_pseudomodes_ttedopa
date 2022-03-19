#!/usr/bin/julia

using LaTeXStrings
using ProgressMeter
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
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")

# Questo programma calcola l'evoluzione della catena di spin isolata,
# usando le tecniche dei MPS ed MPO.

let
  n_sites = 10
  ελ = [(1,0), (100,1), (10,1), (1,1), (0.01,1), (0,1)]

  h1 = sum([reduce(⊗, [i == n ? σᶻ : I₂ for i ∈ 1:n_sites];
                   init=1.0)
            for n ∈ 1:n_sites];
           init=zeros(2^n_sites, 2^n_sites))
  h2 = sum([reduce(⊗, [i == n ? σ⁺⊗σ⁻ + σ⁻⊗σ⁺ : I₂ for i ∈ 1:n_sites-1];
                   init=1.0)
            for n ∈ 1:n_sites-1];
           init=zeros(2^n_sites, 2^n_sites))
  fullH(ε,λ) = -0.5λ * h2 + 0.5ε * h1
  output = Dict()
  for (ε,λ) ∈ ελ
    push!(output, Symbol("$ε/$λ") => eigvals(fullH(ε,λ)))
  end
  CSV.write("spectrum.dat", DataFrame(output))
  return
end
