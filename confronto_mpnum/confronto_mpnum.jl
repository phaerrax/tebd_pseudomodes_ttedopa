#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using CSV

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/operators.jl")

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

  # Leggo il file con i risultati di Python
  pythoncsv = CSV.File(root_path * "/confronto_mpnum/python_output.csv")

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  occ_n_super = []
  snapshot_comp_super = []

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
    n_sites = parameters["number_of_spin_sites"] 
    # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
    sites = siteinds("S=1/2", n_sites)

    # Costruzione dell'operatore di evoluzione
    # ========================================
    localcfs = repeat([ε], n_sites)
    interactioncfs = repeat([1], n_sites-1)
    hlist = twositeoperators(sites, localcfs, interactioncfs)
    #
    function links_odd(τ)
      return [exp(-im * τ * h) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(-im * τ * h) for h in hlist[2:2:end]]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             time_step,
                             parameters["TS_expansion_order"])

    # Simulazione
    # ===========
    # Determina lo stato iniziale a partire dalla stringa data nei parametri
    current_state = parse_init_state(sites,
                                     parameters["chain_initial_state"])

    # Misuro le osservabili sullo stato iniziale
    occ_n = [expect(current_state, "N")]

    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state;
                            cutoff=max_err,
                            maxdim=max_dim)
      push!(occ_n, expect(current_state, "N"))
      next!(progress)
    end

    snapshot_jl = occ_n[end]
    snapshot_py = [last(pythoncsv["occ_n$N"])
                   for N ∈ 1:n_sites]

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(snapshot_comp_super, [snapshot_jl snapshot_py])

  end

  # Grafici
  # =======
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
  savefig(plt, "occ_n.png")

  # Confronto con Python dell'andamento del primo sito
  # --------------------------------------------------
  data_super = []
  for occ_n ∈ occ_n_super
    data = []
    for (row, py_n) ∈ zip(occ_n, pythoncsv["occ_n1"])
      push!(data, row[begin] - py_n)
    end
    push!(data_super, data)
  end

  plt = plot_time_series(data_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["ITensors - mpnum"],
                         linestyles=[:solid],
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_1\rangle",
                         plot_title="Discrepanza tra ITensors e mpnum",
                         plot_size=plot_size
                        )
  savefig(plt, "occ_n1_confronto.png")

  # Istantanea dei numeri di occupazione alla fine
  # ----------------------------------------------
  plt = plot_standalone(snapshot_comp_super,
                        parameter_lists;
                        labels=["ITensors", "mpnum"],
                        linestyles=[:solid, :solid],
                        x_label="Sito",
                        y_label="Numero di occupazione",
                        plot_title="Contronfo tra i numeri di occupazione alla fine",
                        plot_size=plot_size
                        )
  savefig(plt, "snapshot_confronto.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
