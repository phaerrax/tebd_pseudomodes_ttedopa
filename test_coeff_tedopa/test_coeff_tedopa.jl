#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV
using JSON

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/tedopa.jl")
include(lib_path * "/operators.jl")

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
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    Ω = parameters["spectral_density_peak"]
    T = parameters["temperature"]
    γ = parameters["spectral_density_half_width"]
    ωc = parameters["frequency_cutoff"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]
      n_spin_sites = parameters["number_of_spin_sites"]
      n_osc_left = parameters["number_of_oscillators_left"]
      n_osc_right = parameters["number_of_oscillators_right"]
      sites = [siteinds("Osc", n_osc_left; dim=osc_dim);
               siteinds("S=1/2", n_spin_sites);
               siteinds("Osc", n_osc_right; dim=osc_dim)]

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    J(ω) = γ/π * (1 / (γ^2 + (ω-Ω)^2) - 1 / (γ^2 + (ω+Ω)^2))
    Jtherm = ω -> thermalisedJ(J, ω, T, (-ωc, 2ωc))
    Jzero  = ω -> thermalisedJ(J, ω, 0, (0, ωc))
    (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(Jtherm,
                                        (-ωc, 2ωc),
                                        ωc,
                                        n_osc_left-1;
                                        Nquad=nquad,
                                        discretization=lanczos)
    (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(Jzero,
                                        (0, ωc),
                                        ωc,
                                        n_osc_right-1;
                                        Nquad=nquad,
                                        discretization=lanczos)
	gui(plot(Jtherm, -ωc:0.01:2ωc))
    # Raccolgo i coefficienti in due array (uno per quelli a sx, l'altro per
    # quelli a dx) per poterli disegnare assieme nei grafici.
    # (I coefficienti κ sono uno in meno degli Ω! Per ora pareggio le lunghezze
    # inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ωₗ [0; κₗ]]
    osc_chain_coefficients_right = [Ωᵣ [0; κᵣ]]
    
    pyplot()
    p = plot(osc_chain_coefficients_left, label=[L"\Omega" L"\kappa"], title="sx", reuse=false)
    gui(p)
    q = plot(osc_chain_coefficients_right, label=[L"\Omega" L"\kappa"], title="dx", reuse=false)
	gui(q)
    #=
  # Grafico dei coefficienti della chain map
  # ----------------------------------------
  osc_sites = [reverse(1:length(chain[:,1]))
               for chain in osc_chain_coefficients_left_super]
  plt = groupplot(osc_sites,
                  osc_chain_coefficients_left_super,
                  parameter_lists;
                  labels=[L"\Omega_i" L"\kappa_i"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                             "oscillatori (sx)",
                  plotsize=plotsize)

  savefig(plt, "osc_left_coefficients.png")

  osc_sites = [1:length(chain[:,1])
               for chain in osc_chain_coefficients_right_super]
  plt = groupplot(osc_sites,
                  osc_chain_coefficients_right_super,
                  parameter_lists;
                  labels=[L"\Omega_i" L"\kappa_i"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                            "oscillatori (dx)",
                  plotsize=plotsize)

  savefig(plt, "osc_right_coefficients.png")
  =#
  return
end
