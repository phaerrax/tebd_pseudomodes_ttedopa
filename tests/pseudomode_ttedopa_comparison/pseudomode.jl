#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
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
include(joinpath(libpath, "spin_chain_space.jl"))
include(joinpath(libpath, "harmonic_oscillator_space.jl"))
include(joinpath(libpath, "operators.jl"))

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

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

  timesteps_super = []
  occn_PM_super = []
  normalisation_PM_super = []
  range_spins_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["oscillator_spin_interaction_coefficient"]
    γ = parameters["oscillator_damping_coefficient"]
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    range_spins = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=osc_dim)]

    #= Definizione degli operatori nell'equazione di Lindblad
    ======================================================
    I siti del sistema sono numerati come segue:
    | 1 | 2 | ... | n_spin_sites | n_spin_sites+1 | n_spin_sites+2 |
    ↑   │                        │          ↑
    │   └───────────┬────────────┘          │
    │               │                       │
    │        catena di spin                 │
    oscillatore sx                    oscillatore dx
    =#
    localcfs = [ω; repeat([ε], n_spin_sites); ω]
    interactioncfs = [κ; repeat([1], n_spin_sites-1); κ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += γ * (op("Damping", sites[begin]; ω=ω, T=T) *
                         op("Id", sites[begin+1]))
    ℓlist[end] += γ * (op("Id", sites[end-1]) *
                       op("Damping", sites[end]; ω=ω, T=0))
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # Osservabili da misurare
    # =======================
    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                   for n ∈ range_spins]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    ρ₀ = chain(parse_init_state_osc(sites[1], "thermal"; ω=ω, T=T),
               parse_init_state(sites[range_spins], "empty"),
               parse_init_state_osc(sites[end], "empty"))

    # Osservabili
    # -----------
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout,
    normalisation,
    occnlist,
    ranks = evolve(ρ₀,
                   time_step_list,
                   parameters["skip_steps"],
                   parameters["TS_expansion_order"],
                   links_odd,
                   links_even,
                   parameters["MP_compression_error"],
                   parameters["MP_maximum_bond_dimension"];
                   fout=[trace,
                         occn,
                         linkdims])

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occn_PM = mapreduce(permutedims, vcat, occnlist)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    for n ∈ eachindex(range_spins)
      sym = Symbol("occn_spin_PM_$n")
      push!(dict, sym => occn_PM[:, n])
    end
    push!(dict, :norm_PM => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat~1")
    CSV.write(filename, table)

    push!(timesteps_super, tout)
    push!(occn_PM_super, occn_PM)
    push!(normalisation_PM_super, normalisation)
    push!(range_spins_super, range_spins)
  end

  # Grafici
  # =======
  plotsize = (600, 400)

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  plt = groupplot(timesteps_super,
                  occn_PM_super,
                  parameter_lists;
                  labels=[reduce(hcat,
                                 ["S$n" for n ∈ eachindex(range_spins)])
                          for range_spins ∈ range_spins_super],
                  linestyles=[reduce(hcat,
                                     repeat([:solid], length(range_spins)))
                              for range_spins ∈ range_spins_super],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione degli spin (pseudomodi)",
                  plotsize=plotsize)
  savefig(plt, "occn_spins_PM.png")
  
  # Grafico della norma dello stato
  # -------------------------------
  plt = unifiedplot(timesteps_super,
                    normalisation_PM_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\tr\rho(t)",
                    plottitle="Norma dello stato (pseudomodi)",
                    plotsize=plotsize)
  savefig(plt, "normalisation_PM.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
