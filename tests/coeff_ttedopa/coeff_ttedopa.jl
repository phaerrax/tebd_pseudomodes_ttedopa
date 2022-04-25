#!/usr/bin/julia

using LaTeXStrings
using Base.Filesystem
using DataFrames
using CSV
using JSON

rootdirname = "simulazioni_tesi"
sourcepath = Base.source_path()
# Cartella base: determina il percorso assoluto del file in esecuzione, e
# rimuovi tutto ciò che segue rootdirname.
ind = findfirst(rootdirname, sourcepath)
rootpath = sourcepath[begin:ind[end]]
# `rootpath` è la cartella principale del progetto.
libpath = joinpath(rootpath, "lib")

include(joinpath(libpath, "utils.jl"))
include(joinpath(libpath, "plotting.jl"))
include(joinpath(libpath, "tedopa.jl"))

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let
  parameters = Dict("filename" => ARGS[1])
  open(ARGS[1]) do input
    s = JSON.read(input, String)
    # Aggiungo anche il nome del file alla lista di parametri.
    parameters = merge(parameters, JSON.parse(s))
  end

  # Impostazione dei parametri
  # ==========================
  # - parametri tecnici
  nquad = Int(parameters["PolyChaos_nquad"])
  ωc = parameters["frequency_cutoff"]
  n_osc_left = parameters["number_of_oscillators_left"]
  n_osc_right = parameters["number_of_oscillators_right"]

  # - parametri fisici
  Ω = parameters["spectral_density_peak"]
  T = parameters["temperature"]
  γ = parameters["spectral_density_half_width"]
  κ = parameters["spectral_density_overall_factor"]

  # Calcolo dei coefficienti dalla densità spettrale
  J(ω) = κ^2 * 0.5γ/π * (1 / ((0.5γ)^2 + (ω-Ω)^2) - 1 / ((0.5γ)^2 + (ω+Ω)^2))
  Jtherm = ω -> thermalisedJ(J, ω, T)
  Jzero  = ω -> thermalisedJ(J, ω, 0)
  (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(Jtherm,
                                      (-ωc, 0, ωc),
                                      n_osc_left-1;
                                      Nquad=nquad,
                                      discretization=lanczos)
  (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(Jzero,
                                      (0, ωc),
                                      n_osc_right-1;
                                      Nquad=nquad,
                                      discretization=lanczos)

  osc_chain_coefficients_left = [Ωₗ [ηₗ; κₗ]]
  osc_chain_coefficients_right = [Ωᵣ [ηᵣ; κᵣ]]

  pyplot()
  p = plot(osc_chain_coefficients_left, label=[L"\Omega" L"\kappa"], title="sx", reuse=false)
  gui(p)
  q = plot(osc_chain_coefficients_right, label=[L"\Omega" L"\kappa"], title="dx", reuse=false)
  gui(q)

  println(osc_chain_coefficients_left[:,1])
  println(osc_chain_coefficients_left[:,2])
  println(osc_chain_coefficients_right[:,1])
  println(osc_chain_coefficients_right[:,2])
  return
end
