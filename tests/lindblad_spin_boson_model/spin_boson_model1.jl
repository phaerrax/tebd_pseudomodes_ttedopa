#!/usr/bin/julia

using ITensors, DataFrames, LaTeXStrings, CSV
using PseudomodesTTEDOPA

disablegrifqtech()

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
    ω = parameters["oscillator_frequency"]
    if (haskey(parameters, "oscillator_damping_coefficient_left") &&
        haskey(parameters, "oscillator_damping_coefficient_right"))
      γₗ = parameters["oscillator_damping_coefficient_left"]
      γᵣ = parameters["oscillator_damping_coefficient_right"]
    elseif haskey(parameters, "oscillator_damping_coefficient")
      γₗ = parameters["oscillator_damping_coefficient"]
      γᵣ = γₗ
    else
      throw(ErrorException("Oscillator damping coefficient not provided."))
    end
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

    # Osservabili da misurare
    # =======================
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                   for n ∈ 1:length(sites)]
    full_trace = MPS(sites, "vecId")

    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) / trace(ρ) for N in num_op_list])

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Lo stato iniziale della catena è "spin_initial_state" per lo spin
    # attaccato all'oscillatore, "empty" per gli altri.
    if n_spin_sites >= 2
      ρ₀ = chain(parse_init_state_osc(sites[1],
                                      parameters["left_oscillator_initial_state"];
                                      ω=ω, T=T),
                 parse_spin_state(sites[2],
                                  parameters["spin_initial_state"]),
                 parse_init_state(sites[3:end-1],
                                  "empty"),
                 parse_init_state_osc(sites[end], "empty"))
    else
      ρ₀ = chain(parse_init_state_osc(sites[1],
                                      parameters["left_oscillator_initial_state"];
                                      ω=ω, T=T),
                 parse_spin_state(sites[2],
                                  parameters["spin_initial_state"]),
                 parse_init_state_osc(sites[end], "empty"))
    end

    # Evoluzione temporale
    # --------------------
    @time begin
      tout, normalisation, occnlist, ranks = evolve(ρ₀,
                      time_step_list,
                      parameters["skip_steps"],
                      parameters["TS_expansion_order"],
                      links_odd,
                      links_even,
                      parameters["MP_compression_error"],
                      parameters["MP_maximum_bond_dimension"];
                      fout=[trace, occn, linkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output,
    # e la stampo su un file temporaneo che verrà letto dall'altro script.
    dict = Dict(:time => tout)
    for (j, name) in enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n = 1:n_spin_sites];
                              :occ_n_right])
      push!(dict, name => occnlist[:,j])
    end
    for (j, name) in enumerate([Symbol("bond_dim$n")
                                for n ∈ 1:len-1])
      push!(dict, name => ranks[:,j])
    end
    push!(dict, :full_trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat.tmp")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
