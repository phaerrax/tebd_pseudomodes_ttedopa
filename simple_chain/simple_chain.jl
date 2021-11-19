#!/usr/bin/julia

using ITensors
using Plots
using LaTeXStrings
using JSON

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")

# Questo programma calcola l'evoluzione della catena di spin isolata,
# usando le tecniche dei MPS ed MPO.

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
  occ_n_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)

    # Costruzione della catena
    # ========================
    n_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
    # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
    sites = siteinds("S=1/2", n_sites)

    # Stati di singola eccitazione
    single_ex_states = [single_ex_state(sites, j) for j = 1:n_sites]

    # Nuovi operatori di ITensors
    # ===========================
    # Identità su S=1/2
    ITensors.op(::OpName"id", ::SiteType"S=1/2") = [
      1 0
      0 1
    ]

    # Costruzione dell'operatore di evoluzione
    # ========================================
    # In questo sistema semplice non sto usando la matrice identità e l'equazione
    # di Lindblad per calcolare la traiettoria del sistema, quindi non mi servono
    # le funzioni definite nei vari file ausiliari.
    # Uso l'espansione di Trotter-Suzuki al 2° ordine.
    #
    # Ricorda:
    # - gli h_loc agli estremi della catena vanno trattati separatamente
    # - Sz è 1/2*sigma_z, la matrice sigma_z si chiama id, e così per le
    #   altre matrici di Pauli
    links_odd = ITensor[]
    s1 = sites[1]
    s2 = sites[2]
    h_loc = ε * op("Sz", s1) * op("id", s2) +
            1/2 * ε * op("id", s1) * op("Sz", s2) +
            -1/2 * op("S+", s1) * op("S-", s2) +
            -1/2 * op("S-", s1) * op("S+", s2)
    push!(links_odd, exp(-0.5im * time_step * h_loc))
    for j = 3:2:n_sites-3
      s1 = sites[j]
      s2 = sites[j+1]
      h_loc = 1/2 * ε * op("Sz", s1) * op("id", s2) +
              1/2 * ε * op("id", s1) * op("Sz", s2) +
              -1/2 * op("S+", s1) * op("S-", s2) +
              -1/2 * op("S-", s1) * op("S+", s2)
      push!(links_odd, exp(-0.5im * time_step * h_loc))
    end
    s1 = sites[end-1] # j = n_sites-1
    s2 = sites[end] # j = n_sites
    h_loc = 1/2 * ε * op("Sz", s1) * op("id", s2) +
            ε * op("id", s1) * op("Sz", s2) +
            -1/2 * op("S+", s1) * op("S-", s2) +
            -1/2 * op("S-", s1) * op("S+", s2)
    push!(links_odd, exp(-0.5im * time_step * h_loc))

    links_even = ITensor[]
    for j = 2:2:n_sites-2
      s1 = sites[j]
      s2 = sites[j+1]
      h_loc = 1/2 * ε * op("Sz", s1) * op("id", s2) +
              1/2 * ε * op("id", s1) * op("Sz", s2) +
              -1/2 * op("S+", s1) * op("S-", s2) +
              -1/2 * op("S-", s1) * op("S+", s2)
      push!(links_even, exp(-1.0im * time_step * h_loc))
    end

    time_evolution_oplist = vcat(links_odd, links_even, links_odd)

    # Lo stato iniziale ha un'eccitazione nel primo sito
    current_state = single_ex_states[1]

    # Misuro le osservabili sullo stato iniziale
    occ_n = [[abs2(inner(s, current_state)) for s in single_ex_states]]
    #maxdim_monitor = Int[]

    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    for _ in time_step_list[2:end]
      current_state = apply(time_evolution_oplist, current_state; cutoff=max_err, maxdim=max_dim)
      push!(occ_n,
            [abs2(inner(s, current_state)) for s in single_ex_states])
      #push!(maxdim_monitor, maxlinkdim(current_state))
      next!(progress)
    end

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  plot_size = Int(ceil(sqrt(length(parameter_lists)))) .* (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  len = size(hcat(occ_n_super[begin]...), 1)
  # È la lunghezza delle righe di vari array `occ_n`: per semplicità assumo che
  # siano tutti della stessa forma, altrimenti dovrei far calcolare alla
  # funzione `plot_time_series` anche tutto ciò che varia con tale lunghezza,
  # come `labels` e `linestyles`.
  #
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=string.(1:len),
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione",
                         plot_size=plot_size
                        )
  savefig(plt, "occupation_numbers.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
