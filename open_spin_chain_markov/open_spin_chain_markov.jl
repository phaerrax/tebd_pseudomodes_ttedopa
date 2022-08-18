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
  normalisation_super = []

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
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ 1:n_spin_sites])
                   for n ∈ 1:n_spin_sites]

    # Costruzione dell'operatore di evoluzione
    # ========================================
    localcfs = repeat([ε], n_spin_sites)
    interactioncfs = repeat([1], n_spin_sites-1)
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    avgn(ε,T) = T == 0 ? 0 : (ℯ^(ε/T) - 1)^(-1)
    ξL = κ * (1 + 2avgn(ε,T))
    ξR = κ * (1 + 2avgn(ε,0))
    ℓlist[begin] += ξL * op("Damping", sites[begin]) * op("Id", sites[begin+1])
    ℓlist[end] += ξR * op("Id", sites[end-1]) * op("Damping", sites[end])
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # Osservabili da misurare
    # =======================
    # - la corrente di spin
    current_adjsitesops = [current(sites, j, j+1)
                           for j ∈ eachindex(sites)[1:end-1]]
                    
    # - la traccia di ρ
    full_trace = MPS(sites, "vecId")

    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)
    spincurrent(ρ) = real.([inner(j, ρ)
                        for j ∈ current_adjsitesops]) ./ trace(ρ)

    # Simulazione
    # ===========
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    ρ₀ = parse_init_state(sites, parameters["chain_initial_state"])

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout,
      normalisation,
      occnlist,
      current_adjsiteslist,
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
                           spincurrent,
                           linkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    spincurrentlist = mapreduce(permutedims, vcat, current_adjsiteslist)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    sitelabels = string.("S", 1:n_spin_sites)
    for (n, label) ∈ enumerate(sitelabels)
      push!(dict, Symbol(string("occn_", label)) => occnlist[:, n])
    end
    for n ∈ 1:n_spin_sites-1
      from = sitelabels[n]
      to   = sitelabels[n + 1]
      sym  = "current_adjsites_$from/$to"
      push!(dict, Symbol(sym) => spincurrentlist[:, n])
      sym  = "rank_$from/$to"
      push!(dict, Symbol(sym) => ranks[:, n])
    end
    push!(dict, :norm => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occ_n_super, occnlist)
    push!(current_adjsites_super, spincurrentlist)
    push!(bond_dimensions_super, ranks)
    push!(normalisation_super, normalisation)
  end

  # Grafici
  # =======
  
  plotsize = (600, 400)

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_super[begin], 2)
  plt = groupplot(timesteps_super,
                  occ_n_super,
                  parameter_lists;
                  labels=reduce(hcat, string.(1:N)),
                  linestyles=reduce(hcat, repeat([:solid], N)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione",
                  plotsize=plotsize)

  savefig(plt, "occ_n_all.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  N = size(bond_dimensions_super[begin], 2)
  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=reduce(hcat, ["($j,$(j+1))" for j ∈ 1:N]),
                  linestyles=reduce(hcat, repeat([:solid], N)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\chi_{k,k+1}(t)",
                  plottitle="Ranghi del MPS",
                  plotsize=plotsize)

  savefig(plt, "bond_dimensions.png")

  # Grafico della traccia della matrice densità
  # -------------------------------------------
  # Questo serve più che altro per controllare che rimanga sempre pari a 1.
  plt = unifiedplot(timesteps_super,
                    normalisation_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\operatorname{tr}\,\rho(t)",
                    plottitle="Normalizzazione della matrice densità",
                    plotsize=plotsize)

  savefig(plt, "dm_normalisation.png")

  # Grafico della corrente di spin
  # ------------------------------
  N = size(current_adjsites_super[begin], 2)
  plt = groupplot(timesteps_super,
                  current_adjsites_super,
                  parameter_lists;
                  labels=reduce(hcat, ["($j,$(j+1))" for j ∈ 1:N]),
                  linestyles=reduce(hcat, repeat([:solid], N)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"j_{k,k+1}(t)",
                  plottitle="Corrente di spin",
                  plotsize=plotsize)

  savefig(plt, "spin_current.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
