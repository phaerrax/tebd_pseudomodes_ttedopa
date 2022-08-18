#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, Plots, QuadGK
using PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma calcola l'energia di ripartizione relativa residua,
# come metro di misura della bontà del cutoff scelto.

let
  parameter_lists = load_parameters(ARGS)
  tot_sim_n = length(parameter_lists)

  # Se il primo argomento da riga di comando è una cartella (che dovrebbe
  # contenere i file dei parametri), mi sposto subito in tale posizione in modo
  # che i file di output, come grafici e tabelle, siano salvati insieme ai file
  # di parametri.
  prev_dir = pwd()
  if isdir(ARGS[1])
    cd(ARGS[1])
  end

  cutoffs = []
  residuals = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================
    # - parametri tecnici
    nquad = Int(parameters["PolyChaos_nquad"])
    # parameters["frequency_cutoff"] viene ignorato
    nosc = parameters["number_of_oscillators_left"]

    # - parametri fisici
    Ω₀ = parameters["spectral_density_peak"]
    T = parameters["temperature"]
    γ = parameters["spectral_density_half_width"]
    κ = parameters["spectral_density_overall_factor"]

    # Calcolo dei coefficienti dalla densità spettrale
    Θ(x) = x ≥ 0 ? 1 : 0
    thermf(T,ω) = T == 0 ? Θ(ω) : 0.5*(coth(0.5ω/T) + 1)
    J(T,ω) = thermf(T,ω) * κ^2 * 0.5γ/π * (hypot(0.5γ, ω-Ω₀)^(-2) - hypot(0.5γ, ω+Ω₀)^(-2))

    cutofflist = collect(range(start=Ω₀, stop=10Ω₀, length=1000))
    resrelenergy = [quadgk(ω -> J(T,ω)/ω, ωc, Inf)[1] / quadgk(ω -> J(T,ω)/ω, 0, Inf)[1] for ωc ∈ cutofflist]

    push!(cutoffs, cutofflist)
    push!(residuals, resrelenergy)

    dict = Dict(:cutoff => cutofflist)
    push!(dict, :residual_reorg_energy => resrelenergy)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    CSV.write(filename, table)
  end

  plotsize = (600, 400)

  plt = unifiedplot(cutoffs,
                    [clamp!(values, 0, 1e-4) for values ∈ residuals],
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\omega_c",
                    ylabel="Residual",
                    plottitle="Neglected relative reorganisation energy",
                    plotsize=plotsize)
  savefig(plt, "residual.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
end
