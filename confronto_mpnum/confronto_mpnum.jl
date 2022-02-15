#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
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

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  occ_n_super = []
  occ_n_comp_super = []
  snapshot_comp_super = []
  ranks_super = []
  timesteps_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    n_sites = parameters["number_of_spin_sites"] 
    ε = parameters["spin_excitation_energy"]
    # λ = 1

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)

    # Leggo il file con i risultati di Python
    pythoncsv = CSV.File(replace(parameters["filename"],
                                 ".json" => "_python.csv"))
    occ_n_py = hcat([pythoncsv["occ_n$N"] for N∈1:n_sites]...)

    # Costruzione della catena
    # ========================
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
    nums = expect(current_state, "N")
    occ_n = [nums]
    occ_n_comp = [nums - occ_n_py[1,:]]
    ranks = [linkdims(current_state)]

    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    for (j, _) in zip(2:length(time_step_list), time_step_list[2:end])
      current_state = apply(evo,
                            current_state;
                            cutoff=max_err,
                            maxdim=max_dim)
      nums = expect(current_state, "N")
      push!(occ_n, nums)
      push!(occ_n_comp, nums - occ_n_py[j,:])
      push!(ranks, linkdims(current_state))
      next!(progress)
    end

    snapshot_jl = occ_n[end]
    snapshot_py = [last(pythoncsv["occ_n$N"])
                   for N ∈ 1:n_sites]

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list)
    push!(occ_n_super, hcat(occ_n...)')
    push!(occ_n_comp_super, hcat(occ_n_comp...)')
    push!(snapshot_comp_super, [snapshot_jl snapshot_py])
    push!(ranks_super, hcat(ranks...)')
  end

  # Grafici
  # =======
  plotsize = (600, 400)

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_super[begin])[2]
  plt = groupplot(timesteps_super,
                  occ_n_super,
                  parameter_lists;
                  labels=hcat(string.(1:N)...),
                  linestyles=hcat(repeat([:solid], N...)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione",
                  plotsize=plotsize)
                  
  savefig(plt, "occ_n.png")

  # Confronto con Python dell'andamento del primo sito
  # --------------------------------------------------
  cols = [1, 5, 10]
  selection = [mat[:, cols] for mat in occ_n_comp_super]
  plt = groupplot(timesteps_super,
                  selection,
                  parameter_lists;
                  labels=hcat(string.(cols)...),
                  linestyles=hcat(repeat([:solid], length(cols))...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i\rangle",
                  plottitle="Discrepanza tra ITensors e mpnum",
                  plotsize=plotsize)
                  
  savefig(plt, "discrepanza.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  N = size(ranks_super[begin])[2]
  plt = groupplot(timesteps_super,
                  ranks_super,
                  parameter_lists;
                  labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\chi_{k,k+1}(t)",
                  plottitle="Ranghi del MPS",
                  plotsize=plotsize)

  savefig(plt, "bond_dimensions.png")

  # Istantanea dei numeri di occupazione alla fine
  # ----------------------------------------------
  plt = groupplot([1:size(arr, 1) for arr in snapshot_comp_super],
                  snapshot_comp_super,
                  parameter_lists;
                  labels=["ITensors" "mpnum"],
                  linestyles=[:solid :solid],
                  commonxlabel="Sito",
                  commonylabel="Numero di occupazione",
                  plottitle="Contronfo tra i numeri di occupazione alla fine",
                  plotsize=plotsize)

  savefig(plt, "snapshot_confronto.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
