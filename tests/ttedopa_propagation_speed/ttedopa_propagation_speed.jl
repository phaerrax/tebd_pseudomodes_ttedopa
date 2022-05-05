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

# Questo programma mostra come evolve un'eccitazione solitaria in una catena
# di oscillatori ottenuta con il T-TEDOPA, in modo da determinare una lunghezza
# ottimale per la catena nel resto delle simulazioni.
# Confronterò l'evoluzione di un sistema con i coefficienti T-TEDOPA con quella
# di un sistema in cui ωₖ e κₖ sono sostituiti dai loro valori asintotici.

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

  timeranges = []
  chainranges = []
  frequencies = []
  couplingcoeffs = []
  first2sitesevos = []
  firstsiteevos = []
  asymptoticevos = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================
    # - parametri tecnici
    nquad = Int(parameters["PolyChaos_nquad"])
    ωc = parameters["frequency_cutoff"]
    nosc = parameters["number_of_oscillators_left"]

    # - parametri fisici
    Ω = parameters["spectral_density_peak"]
    T = parameters["temperature"]
    γ = parameters["spectral_density_half_width"]
    κ = parameters["spectral_density_overall_factor"]

    # Calcolo dei coefficienti dalla densità spettrale
    Θ(x) = x ≥ 0 ? 1 : 0
    thermf(T,ω) = T == 0 ? Θ(ω) : 0.5(coth(0.5ω/T) + 1)
    J(T,ω) = thermf(T,ω) * κ^2 * 0.5γ/π * (1 / ((0.5γ)^2 + (ω-Ω)^2) - 1 / ((0.5γ)^2 + (ω+Ω)^2))
    if T != 0
      (Ω, κ, η) = chainmapcoefficients(ω -> J(T,ω),
                                       (-ωc, 0, ωc),
                                       nosc-1;
                                       Nquad=nquad,
                                       discretization=lanczos)
    else
      (Ω, κ, η) = chainmapcoefficients(ω -> J(T,ω),
                                       (0, ωc),
                                       nosc-1;
                                       Nquad=nquad,
                                       discretization=lanczos)
    end
    
    push!(chainranges, 1:nosc)
    push!(frequencies, Ω)
    push!(couplingcoeffs, κ)

    H(ωs, κs) = diagm(0 => ωs) + diagm(-1 => κs) + diagm(1 => κs)
    U(τ,H) = exp(-im * τ * H)

    s₁ = zeros(ComplexF64, nosc)
    s₁[1] = one(eltype(s₁))
    s₂ = zeros(ComplexF64, nosc)
    s₂[2] = one(eltype(s₂))
    
    occn1(ψ) = abs2(dot(s₁, ψ))
    occn2(ψ) = abs2(dot(s₂, ψ))

    t = 0:(parameters["simulation_time_step"]):(parameters["simulation_end_time"])
    ureal = U(step(t), H(Ω,κ))
    uasym = U(step(t), H(repeat([Ω[end]], nosc), repeat([κ[end]], nosc-1)))
    nreal1 = similar(t)
    nreal2 = similar(t)
    nasym1 = similar(t)
    nasym2 = similar(t)
    ψasym = s₁
    ψreal = s₁
    for i ∈ eachindex(t)
      nreal1[i], nreal2[i] = occn1(ψreal), occn2(ψreal)
      nasym1[i], nasym2[i] = occn1(ψasym), occn2(ψasym)
      ψreal = ureal*ψreal
      ψasym = uasym*ψasym
    end
    push!(timeranges, t)
    push!(firstsiteevos, nreal1)
    push!(first2sitesevos, nreal1 .+ nreal2)
    push!(asymptoticevos, nasym1)

    dict = Dict(:time => collect(t))
    push!(dict, :occn_real1 => nreal1)
    push!(dict, :occn_real2 => nreal2)
    push!(dict, :occn_asym1 => nasym1)
    push!(dict, :occn_asym2 => nasym2)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    CSV.write(filename, table)
  end

  plotsize = (600, 400)

  plt = unifiedplot(chainranges,
                    frequencies,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel="k",
                    ylabel="Ωₖ",
                    plottitle="Frequenze T-TEDOPA",
                    plotsize=plotsize)
  savefig(plt, "frequencies.png")

  shorterchainranges = [rn[1:end-1] for rn ∈ chainranges]
  plt = unifiedplot(shorterchainranges,
                    couplingcoeffs,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel="k",
                    ylabel="κₖ",
                    plottitle="Coefficienti di interazione T-TEDOPA",
                    plotsize=plotsize)
  savefig(plt, "couplingcoeffs.png")

  plt = unifiedplot(timeranges,
                    firstsiteevos,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\langle n_1(t)\rangle",
                    plottitle="Evoluzione con coefficienti veri",
                    plotsize=plotsize)
  savefig(plt, "realevos1.png")

  plt = unifiedplot(timeranges,
                    first2sitesevos,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\langle n_1(t)+n_2(t)\rangle",
                    plottitle="Evoluzione con coefficienti veri",
                    plotsize=plotsize)
  savefig(plt, "realevos1+2.png")

  plt = unifiedplot(timeranges,
                    asymptoticevos,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\langle n_1(t)\rangle",
                    plottitle="Evoluzione con coefficienti asintotici",
                    plotsize=plotsize)
  savefig(plt, "asymptoticevos.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
end
