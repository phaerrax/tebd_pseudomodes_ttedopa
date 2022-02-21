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

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/operators.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# Differisce da "chain_with_damped_oscillators" in quanto qui
# indago l'andamento di quantità diverse e lo confronto con una
# soluzione esatta (per un numero di spin sufficientemente basso).

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

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  timesteps_super = []
  occ_n_super = []
  normalisation_super = []
  bond_dimensions_super = []

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
    γₗ = parameters["oscillator_damping_coefficient_left"]
    γᵣ = parameters["oscillator_damping_coefficient_right"]
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
    spin_range = 1 .+ (1:n_spin_sites)

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
    ℓlist[begin] += γₗ * op("Damping", sites[begin]; ω=ω, T=T) *
                         op("Id", sites[begin+1])
    ℓlist[end] += γᵣ * op("Id", sites[end-1]) *
                       op("Damping", sites[end]; ω=ω, T=0)
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             time_step,
                             parameters["TS_expansion_order"])

    # Osservabili da misurare
    # =======================
    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                   for n ∈ 1:length(sites)]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Lo stato iniziale della catena è "spin_initial_state" per lo spin
    # attaccato all'oscillatore, "empty" per gli altri.
    if n_spin_sites >= 2
    current_state = chain(parse_init_state_osc(sites[1],
                                 parameters["left_oscillator_initial_state"];
                                 ω=ω, T=T),
                          parse_spin_state(sites[2],
                                           parameters["spin_initial_state"]),
                          parse_init_state(sites[3:end-1],
                                           "empty"),
                          parse_init_state_osc(sites[end], "empty"))
    else
    current_state = chain(parse_init_state_osc(sites[1],
                                 parameters["left_oscillator_initial_state"];
                                 ω=ω, T=T),
                          parse_spin_state(sites[2],
                                           parameters["spin_initial_state"]),
                          parse_init_state_osc(sites[end], "empty"))
    end

    full_trace = MPS(sites, "vecId")

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = Vector{Real}[[inner(N, current_state) for N in num_op_list]]
    bond_dimensions = Vector{Int}[linkdims(current_state)]
    normalisation = Real[real(inner(full_trace, current_state))]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state,
                            cutoff=max_err,
                            maxdim=max_dim)
      if skip_count % skip_steps == 0
        #=
        Calcolo dapprima la traccia della matrice densità. Se non devia
        eccessivamente da 1, in ogni caso influisce sul valore delle
        osservabili che calcolo successivamente, che si modificano dello
        stesso fattore, e devono essere quindi corrette di un fattore pari
        al reciproco della traccia.
        =#
        trace = real(inner(full_trace, current_state))

        push!(normalisation,
              trace)
        push!(occ_n,
              [real(inner(N, current_state)) for N in num_op_list] ./ trace)
        push!(bond_dimensions,
              linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end
    occ_n_MPS = permutedims(hcat(occ_n...))

    # Creo una tabella con i dati rilevanti da scrivere nel file di output,
    # e la stampo su un file temporaneo che verrà letto dall'altro script.
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    for (j, name) in enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n ∈ 1:n_spin_sites];
                              :occ_n_right])
      push!(dict, name => occ_n_MPS[:,j])
    end
    for (j, name) in enumerate([Symbol("bond_dim$n")
                                for n ∈ 1:n_spin_sites+1])
      push!(dict, name => hcat(bond_dimensions...)[j,:])
    end
    push!(dict, :full_trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat.tmp")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list[1:skip_steps:end])
    push!(occ_n_super, occ_n_MPS)
    push!(bond_dimensions_super, permutedims(hcat(bond_dimensions...)))
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return

