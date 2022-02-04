#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV

# Se lo script viene eseguito su Qtech, devo disabilitare l'output
# grafico altrimenti il programma si schianta.
if gethostname() == "qtech.fisica.unimi.it"
  ENV["GKSwstype"] = "100"
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

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  timesteps_super = []
  occ_n_super = []
  spin_current_super = []
  bond_dimensions_super = []
  chain_levels_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["spin_damping_coefficient"]
    T = parameters["temperature"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
    # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
    sites = siteinds("vecS=1/2", n_spin_sites)

    # Stati di singola eccitazione
    single_ex_states = [single_ex_state(sites, j) for j = 1:n_spin_sites]

    # Costruzione dell'operatore di evoluzione
    # ========================================
    localcfs = repeat([ε], n_spin_sites)
    interactioncfs = repeat([1], n_spin_sites-1)
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ξL = T==0 ? κ : κ * (1 + 2 / (ℯ^(ε/T) - 1))
    ξR = κ
    ℓlist[begin] += ξL * op("Damping", sites[begin]) * op("Id", sites[begin+1])
    ℓlist[end] += ξR * op("Id", sites[end-1]) * op("Damping", sites[end])
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
    # - la corrente di spin
    spin_current_ops = spin_current_op_list(sites)

    # - l'occupazione degli autospazi dell'operatore numero
    # Ad ogni istante proietto lo stato corrente sugli autostati
    # dell'operatore numero della catena di spin, vale a dire calcolo
    # tr(ρₛ Pₙ) dove ρₛ è la matrice densità ridotta della catena di spin
    # e Pₙ è il proiettore ortogonale sull'n-esimo autospazio di N
    num_eigenspace_projs = [level_subspace_proj(sites, n) for n=0:n_spin_sites]

    # Simulazione
    # ===========
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    current_state = parse_init_state(sites, parameters["chain_initial_state"])

    # Misuro le osservabili sullo stato iniziale
    occ_n = [[inner(s, current_state) for s in single_ex_states]]
    bond_dimensions = [linkdims(current_state)]
    spin_current = [[real(inner(j, current_state)) for j in spin_current_ops]]
    chain_levels = [levels(num_eigenspace_projs, current_state)]

    # ...e si parte!
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state,
                            cutoff=max_err,
                            maxdim=max_dim)
      #
      if skip_count % skip_steps == 0
        push!(occ_n,
              [real(inner(s, current_state)) for s in single_ex_states])
        push!(spin_current,
              [real(inner(j, current_state)) for j in spin_current_ops])
        push!(chain_levels,
              levels(num_eigenspace_projs, current_state))
        push!(bond_dimensions,
              linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([Symbol("occ_n_spin$n") for n = 1:n_spin_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(spin_current...)
    for (j, name) in enumerate([Symbol("spin_current$n") for n = 1:n_spin_sites-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(chain_levels...)
    for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(bond_dimensions...)
    for (j, name) in enumerate([Symbol("bond_dim$n") for n = 1:n_spin_sites-1])
      push!(dict, name => tmp_list[j,:])
    end
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list[1:skip_steps:end])
    push!(occ_n_super, permutedims(hcat(occ_n...)))
    push!(spin_current_super, permutedims(hcat(spin_current...)))
    push!(chain_levels_super, permutedims(hcat(chain_levels...)))
    push!(bond_dimensions_super, permutedims(hcat(bond_dimensions...)))
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
                  labels=["L" string.(1:N-2)... "R"],
                  linestyles=[:dash repeat([:solid], N-2)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione",
                  plotsize=plotsize)

  savefig(plt, "occ_n_all.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  N = size(bond_dimensions_super[begin])[2]
  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\chi_{k,k+1}(t)",
                  plottitle="Ranghi del MPS",
                  plotsize=plotsize)

  savefig(plt, "bond_dimensions.png")

  # Grafico della corrente di spin
  # ------------------------------
  N = size(spin_current_super[begin])[2]
  plt = groupplot(timesteps_super,
                  spin_current_super,
                  parameter_lists;
                  labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"j_{k,k+1}(t)",
                  plottitle="Corrente di spin",
                  plotsize=plotsize)

  savefig(plt, "spin_current.png")

  # Grafico dell'occupazione degli autospazi di N della catena di spin
  # ------------------------------------------------------------------
  # L'ultimo valore di ciascuna riga rappresenta la somma di tutti i
  # restanti valori.
  N = size(chain_levels_super[begin])[2] - 1
  plt = groupplot(timesteps_super,
                  chain_levels_super,
                  parameter_lists;
                  labels=[string.(0:N-1)... "total"],
                  linestyles=[repeat([:solid], N)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"n(",
                  plottitle="Occupazione degli autospazi "
                  * "della catena di spin",
                  plotsize=plotsize)

  savefig(plt, "chain_levels.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
