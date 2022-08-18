#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, Plots
using PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let  
  @info "Lettura dei file con i parametri."
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
  current_adjsites_super = []
  bond_dimensions_super = []
  chain_levels_super = []
  osc_levels_left_super = []
  osc_levels_right_super = []
  normalisation_super = []

  # Precaricamento
  # ==============
  # Se in tutte le liste di parametri il numero di siti è lo stesso, posso
  # definire qui una volta per tutte alcuni elementi "pesanti" che servono dopo.
  n_spin_sites_list = [p["number_of_spin_sites"]
                       for p ∈ parameter_lists]
  kappa_list = [p["oscillator_spin_interaction_coefficient"]
                for p ∈ parameter_lists]
  # I file dei parametri potrebbero avere `cold_...` e `hot_...` oppure soltanto
  # `oscillator_space_dimension`. Devo differenziare i due casi.
  # Se cerco di costruire la lista dei `cold_...` ma nei file non c'è questa
  # chiave, Julia lancia un KeyError.
  # Eseguo quindi in questo punto del codice quello che faccio poi con γ:
  # controllo se esistono i due oscdim differenti, altrimenti prendo quello
  # in comune e lo copio nei due valori. Se non c'è neanche quello, va bene che
  # Julia lanci un errore: significa che i file di parametri sono malformati.
  for p ∈ parameter_lists
    if (!haskey(p, "hot_oscillator_space_dimension") &&
        !haskey(p, "cold_oscillator_space_dimension"))
      push!(p, "hot_oscillator_space_dimension" => p["oscillator_space_dimension"])
      push!(p, "cold_oscillator_space_dimension" => p["oscillator_space_dimension"])
    end
  end
  hotoscdim_list = [p["hot_oscillator_space_dimension"]
                    for p ∈ parameter_lists]
  coldoscdim_list = [p["cold_oscillator_space_dimension"]
                    for p ∈ parameter_lists]
  if (allequal(n_spin_sites_list) &&
      allequal(kappa_list) &&
      allequal(hotoscdim_list) &&
      allequal(coldoscdim_list))
    preload = true
    hotoscdim = first(hotoscdim_list)
    coldoscdim = first(coldoscdim_list)
    n_spin_sites = first(n_spin_sites_list)
    κ = first(kappa_list)

    spin_range = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=hotoscdim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=coldoscdim)]


    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                      [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                  for n ∈ 1:length(sites)]

    # - la corrente tra siti
    current_adjsites_ops = [current(sites, j, j+1)
                             for j ∈ spin_range[1:end-1]]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")
  else
    preload = false
  end

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    @info "($current_sim_n di $tot_sim_n) Costruzione degli operatori di evoluzione temporale."
    # Impostazione dei parametri
    # ==========================

    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["oscillator_spin_interaction_coefficient"]
    Ω = parameters["oscillator_frequency"]
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
    hotoscdim = parameters["hot_oscillator_space_dimension"]
    coldoscdim = parameters["cold_oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    if !preload
      n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
      spin_range = 1 .+ (1:n_spin_sites)

      sites = [siteinds("HvOsc", 1; dim=hotoscdim);
               siteinds("HvS=1/2", n_spin_sites);
               siteinds("HvOsc", 1; dim=coldoscdim)]
    end

    # Definizione degli operatori nell'equazione di Lindblad
    # ======================================================
    localcfs = [Ω; repeat([ε], n_spin_sites); Ω]
    interactioncfs = [κ; repeat([1], n_spin_sites-1); κ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += γₗ * (op("Damping", sites[begin]; ω=Ω, T=T) *
                          op("Id", sites[begin+1]))
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
                        op("Damping", sites[end]; ω=Ω, T=0))
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # Osservabili da misurare
    # =======================
    if !preload
      # - i numeri di occupazione
      num_op_list = [MPS(sites,
                        [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                    for n ∈ 1:length(sites)]

      # - la corrente tra siti
      current_adjsites_ops = [current(sites, j, j+1)
                               for j ∈ spin_range[1:end-1]]

      # - la normalizzazione (cioè la traccia) della matrice densità
      full_trace = MPS(sites, "vecId")
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    @info "($current_sim_n di $tot_sim_n) Creazione dello stato iniziale."
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    ρ₀ = chain(parse_init_state_osc(sites[1],
                                    parameters["left_oscillator_initial_state"];
                                    ω=Ω, T=T),
               parse_init_state(sites[2:end-1],
                                parameters["chain_initial_state"]),
               parse_init_state_osc(sites[end], "empty"))

    # Osservabili
    # -----------
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)
    current_adjsites(ρ) = real.([inner(j, ρ)
                                 for j ∈ current_adjsites_ops]) ./ trace(ρ)

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout,
    normalisation,
    occnlist,
    current_adjsites_list,
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
                         current_adjsites,
                         linkdims])

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    current_adjsites_list = mapreduce(permutedims, vcat, current_adjsites_list)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    @info "($current_sim_n di $tot_sim_n) Creazione delle tabelle di output."
    dict = Dict(:time => tout)
    for (j, name) ∈ enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n ∈ 1:n_spin_sites];
                              :occ_n_right])
      push!(dict, name => occnlist[:,j])
    end
    for j ∈ 1:size(current_adjsites_list, 2)
      sym = Symbol("current_$j/$(j+1)")
      push!(dict, sym => current_adjsites_list[:,j])
    end
    for j ∈ 1:size(ranks, 2)
      sym = Symbol("bond_dim$j/$(j+1)")
      push!(dict, sym => ranks[:,j])
    end
    push!(dict, :trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occ_n_super, occnlist)
    push!(current_adjsites_super, current_adjsites_list)
    push!(bond_dimensions_super, ranks)
    push!(normalisation_super, normalisation)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  @info "Creazione dei grafici."
  plotsize = (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  # Grafico della traccia della matrice densità
  # -------------------------------------------
  # Questo serve più che altro per controllare che rimanga sempre pari a 1.
  plt = unifiedplot(timesteps_super,
                    normalisation_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\mathrm{tr}\,\rho(t)",
                    plottitle="Normalizzazione della matrice densità",
                    plotsize=plotsize)

  savefig(plt, "dm_normalisation.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
