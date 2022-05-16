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
  current_allsites_super = []
  forwardflux_super = []
  backwardflux_super = []
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
      allequal(hotoscdim_list) &&
      allequal(coldoscdim_list) &&
      allequal(kappa_list))
    preload = true
    n_spin_sites = first(n_spin_sites_list)
    hotoscdim = first(hotoscdim_list)
    coldoscdim = first(coldoscdim_list)
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
    current_adjsites_ops = [-2κ*current(sites, 1, 2);
                            [current(sites, j, j+1)
                             for j ∈ spin_range[1:end-1]];
                            -2κ*current(sites, spin_range[end], spin_range[end]+1)]
    current_allsites_ops = [current(sites, i, j)
                            for i ∈ spin_range
                            for j ∈ spin_range
                            if j > i]

    # - l'occupazione degli autospazi dell'operatore numero
    # Ad ogni istante proietto lo stato corrente sugli autostati
    # dell'operatore numero della catena di spin, vale a dire calcolo
    # tr(ρₛ Pₙ) dove ρₛ è la matrice densità ridotta della catena di spin
    # e Pₙ è il proiettore ortogonale sull'n-esimo autospazio di N
    num_eigenspace_projs = [embed_slice(sites,
                                        spin_range,
                                        level_subspace_proj(sites[spin_range], n))
                            for n=0:n_spin_sites]

    # - l'occupazione dei livelli degli oscillatori
    osc_levels_projs_left = [embed_slice(sites,
                                         1:1,
                                         osc_levels_proj(sites[1], n))
                             for n=0:hotoscdim-1]
    osc_levels_projs_right = [embed_slice(sites,
                                          n_spin_sites+2:n_spin_sites+2,
                                          osc_levels_proj(sites[end], n))
                              for n=0:coldoscdim-1]

    # - la normalizzazione (cioè la traccia) della matrice densità
    traceMPS = MPS(sites, "vecId")
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
    ω = parameters["oscillator_frequency"]
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
    ℓlist[begin] += γₗ * (op("Damping", sites[begin]; ω=ω, T=T) *
                          op("Id", sites[begin+1]))
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
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
    if !preload
      # - i numeri di occupazione
      num_op_list = [MPS(sites,
                         [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                     for n ∈ 1:length(sites)]

      # - la corrente tra siti
      current_adjsites_ops = [-2κ*current(sites, 1, 2);
                              [current(sites, j, j+1)
                               for j ∈ spin_range[1:end-1]];
                              -2κ*current(sites, spin_range[end], spin_range[end]+1)]
      current_allsites_ops = [current(sites, i, j)
                              for i ∈ spin_range
                              for j ∈ spin_range
                              if j > i]

      # - l'occupazione degli autospazi dell'operatore numero
      num_eigenspace_projs = [embed_slice(sites,
                                          spin_range,
                                          level_subspace_proj(sites[spin_range], n))
                              for n=0:n_spin_sites]

      # - l'occupazione dei livelli degli oscillatori
      osc_levels_projs_left = [embed_slice(sites,
                                           1:1,
                                           osc_levels_proj(sites[1], n))
                               for n=0:hotoscdim-1]
      osc_levels_projs_right = [embed_slice(sites,
                                            n_spin_sites+2:n_spin_sites+2,
                                            osc_levels_proj(sites[end], n))
                                for n=0:coldoscdim-1]

      # - la normalizzazione (cioè la traccia) della matrice densità
      traceMPS = MPS(sites, "vecId")
    end

    # Memo: [(i,j) for i ∈ 1:n for j ∈ 1:n] =
    # [(1,1) (1,2) … (1,n) (2,1) … (n,n-1) (n,n)]
    forward_flux_ops = [forwardflux(sites, i, j)
                        for i ∈ spin_range
                        for j ∈ spin_range]
    backward_flux_ops = [backwardflux(sites, i, j)
                         for i ∈ spin_range
                         for j ∈ spin_range]
    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    @info "($current_sim_n di $tot_sim_n) Creazione dello stato iniziale."
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    ρ₀ = chain(parse_init_state_osc(sites[1],
                                    parameters["left_oscillator_initial_state"];
                                    ω=ω, T=T),
               parse_init_state(sites[2:end-1],
                                parameters["chain_initial_state"]),
               parse_init_state_osc(sites[end], "empty"))

    # Osservabili
    # -----------
    trace(ρ) = real(inner(traceMPS, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)
    current_adjsites(ρ) = real.([inner(j, ρ)
                                 for j ∈ current_adjsites_ops]) ./ trace(ρ)
    function forwardflux_f(ρ)
      pairs = [(i,j) for i ∈ 1:n_spin_sites for j ∈ 1:n_spin_sites]
      mat = zeros(n_spin_sites, n_spin_sites)
      trρ = trace(ρ)
      for (Ff, (i,j)) in zip(forward_flux_ops, pairs)
        mat[i,j] = real(inner(Ff, ρ) / trρ)
      end
      return Base.vec(mat')
    end
    function backwardflux_f(ρ)
      pairs = [(i,j) for i ∈ 1:n_spin_sites for j ∈ 1:n_spin_sites]
      mat = zeros(n_spin_sites, n_spin_sites)
      trρ = trace(ρ)
      for (Bf, (i,j)) in zip(backward_flux_ops, pairs)
        mat[i,j] = real(inner(Bf, ρ) / trρ)
      end
      return Base.vec(mat')
    end
    function current_allsites(ρ)
      pairs = [(i,j) for i ∈ 1:n_spin_sites for j ∈ 1:n_spin_sites if j > i]
      mat = zeros(n_spin_sites, n_spin_sites)
      trρ = trace(ρ)
      for (J, (k,l)) in zip(current_allsites_ops, pairs)
        mat[k,l] = real(inner(J, ρ) / trρ)
      end
      mat .-= transpose(mat)
      return Base.vec(mat')
    end
    chainlevels(ρ) = real.(inner.(Ref(ρ), num_eigenspace_projs)) ./ trace(ρ)
    osclevelsL(ρ) = real.(inner.(Ref(ρ), osc_levels_projs_left)) ./ trace(ρ)
    osclevelsR(ρ) = real.(inner.(Ref(ρ), osc_levels_projs_right)) ./ trace(ρ)

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout,
      normalisation,
      occnlist,
      current_adjsites_list,
      current_allsites_list,
      forwardflux_list,
      backwardflux_list,
      ranks,
      osclevelsLlist,
      osclevelsRlist,
      chainlevelslist = evolve(ρ₀,
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
                                     current_allsites,
                                     forwardflux_f,
                                     backwardflux_f,
                                     linkdims,
                                     osclevelsL,
                                     osclevelsR,
                                     chainlevels])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    current_adjsites_list = mapreduce(permutedims, vcat, current_adjsites_list)
    current_allsites_list = mapreduce(permutedims, vcat, current_allsites_list)
    forwardflux_list = mapreduce(permutedims, vcat, forwardflux_list)
    backwardflux_list = mapreduce(permutedims, vcat, backwardflux_list)
    ranks = mapreduce(permutedims, vcat, ranks)
    chainlevelslist = mapreduce(permutedims, vcat, chainlevelslist)
    osclevelsLlist = mapreduce(permutedims, vcat, osclevelsLlist)
    osclevelsRlist = mapreduce(permutedims, vcat, osclevelsRlist)

    @info "($current_sim_n di $tot_sim_n) Creazione delle tabelle di output."
    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    push!(dict, :trace => normalisation)
    occn_symbols = [:occ_n_left;
                    [Symbol("occ_n_spin$n") for n ∈ 1:n_spin_sites];
                    :occ_n_right]
    for (n, sym) in enumerate(occn_symbols)
      push!(dict, sym => occnlist[:, n])
    end
    for n ∈ 1:size(current_adjsites_list, 2)
      sym = Symbol("current_adjsites$n")
      push!(dict, sym => current_adjsites_list[:, n])
    end
    current_symbols = [Symbol("current_$i/$j")
                       for i ∈ 1:n_spin_sites
                       for j ∈ 1:n_spin_sites]
    for (n, sym) ∈ enumerate(current_symbols)
        push!(dict, sym => current_allsites_list[:, n])
    end
    for n ∈ 0:hotoscdim-1
      sym = Symbol("levels_left$n")
      push!(dict, sym => osclevelsLlist[:, n+1])
    end
    for n ∈ 0:coldoscdim-1
      sym = Symbol("levels_right$n")
      push!(dict, sym => osclevelsRlist[:, n+1])
    end
    for n ∈ 0:n_spin_sites
      sym = Symbol("levels_chain$n") 
      push!(dict, sym => chainlevelslist[:, n+1])
    end
    for n ∈ 1:size(ranks, 2)
      sym = Symbol("bond_dim_$n/$(n+1)")
      push!(dict, sym => ranks[:, n])
    end
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occ_n_super, occnlist)
    push!(current_adjsites_super, current_adjsites_list)
    push!(current_allsites_super, current_allsites_list)
    push!(forwardflux_super, forwardflux_list)
    push!(backwardflux_super, backwardflux_list)
    push!(chain_levels_super, chainlevelslist)
    push!(osc_levels_left_super, osclevelsLlist)
    push!(osc_levels_right_super, osclevelsRlist)
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

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_super[begin], 2)
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

  # Grafico dell'occupazione del primo oscillatore (riunito)
  # -------------------------------------------------------
  # Estraggo da occ_n_super i valori dell'oscillatore sinistro.
  occ_n_osc_left_super = [occ_n[:,1] for occ_n in occ_n_super]
  plt = unifiedplot(timesteps_super,
                    occ_n_osc_left_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\langle n_L(t)\rangle",
                    plottitle="Occupazione dell'oscillatore sx",
                    plotsize=plotsize)

  savefig(plt, "occ_n_osc_left.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  spinsonly = [mat[:, 2:end-1] for mat in occ_n_super]
  plt = groupplot(timesteps_super,
                  spinsonly,
                  parameter_lists;
                  labels=hcat(string.(1:N-2)...),
                  linestyles=hcat(repeat([:solid], N-2)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (solo spin)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_spins_only.png")

  # Grafico dei numeri di occupazione (oscillatori + totale catena)
  # ---------------------------------------------------------------
  # sum(X, dims=2) prende la matrice X e restituisce un vettore colonna
  # le cui righe sono le somme dei valori sulle rispettive righe di X.
  sums = [[mat[:, 1] sum(mat[:, 2:end-1], dims=2) mat[:, end]]
          for mat in occ_n_super]
  plt = groupplot(timesteps_super,
                  sums,
                  parameter_lists;
                  labels=["L" "catena" "R"],
                  linestyles=[:solid :dot :solid],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (oscillatori + totale catena)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_sums.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  N = size(bond_dimensions_super[begin], 2)
  sitelabels = ["L"; string.(1:N-1); "R"]
  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=reduce(hcat,
                                ["($(sitelabels[j]),$(sitelabels[j+1]))"
                                 for j ∈ eachindex(sitelabels)[1:end-1]]),
                  linestyles=[:dash repeat([:solid], N-2)... :dash],
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
  sitelabels = ["L"; string.(1:N+1); "R"]
  plt = groupplot(timesteps_super,
                  current_adjsites_super,
                  parameter_lists;
                  labels=reduce(hcat,
                                ["($(sitelabels[j]),$(sitelabels[j+1]))"
                                 for j ∈ eachindex(sitelabels)[1:end-1]]),
                  linestyles=reduce(hcat, [:solid for _ ∈ sitelabels]),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle j_{k,k+1}\rangle",
                  plottitle="Corrente (tra siti adiacenti)",
                  plotsize=plotsize)

  savefig(plt, "current_btw_adjacent_sites.png")

  max_n_spins = maximum([p["number_of_spin_sites"] for p in parameter_lists])
  data = [table[:, 1:p["number_of_spin_sites"]]
          for (p, table) in zip(parameter_lists, current_allsites_super)]
  plt = groupplot(timesteps_super,
                  data,
                  parameter_lists;
                  labels=reduce(hcat, ["l=$l" for l ∈ 1:max_n_spins]),
                  linestyles=reduce(hcat, repeat([:solid], max_n_spins)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=LaTeXString("\\langle j_{1,l}\\rangle"),
                  plottitle="Corrente tra spin (dal 1° spin)",
                  plotsize=plotsize)

  savefig(plt, "spin_current_fromsite1.png")

  data = [table[:, 3*p["number_of_spin_sites"] .+ (1:p["number_of_spin_sites"])]
          for (p, table) in zip(parameter_lists, current_allsites_super)]
  plt = groupplot(timesteps_super,
                  data,
                  parameter_lists;
                  labels=reduce(hcat, ["l=$l" for l ∈ 1:max_n_spins]),
                  linestyles=reduce(hcat, repeat([:solid], max_n_spins)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=LaTeXString("\\langle j_{4,l}\\rangle"),
                  plottitle="Corrente tra spin (dal 4° spin)",
                  plotsize=plotsize)

  savefig(plt, "spin_current_fromsite4.png")

  data = [table[:, 1:p["number_of_spin_sites"]]
          for (p, table) in zip(parameter_lists, forwardflux_super)]
  plt = groupplot(timesteps_super,
                  data,
                  parameter_lists;
                  labels=reduce(hcat, ["k=$k" for k ∈ 1:max_n_spins]),
                  linestyles=reduce(hcat, repeat([:solid], max_n_spins)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"1 \to k",
                  plottitle="Flusso di spin in avanti, dal primo sito",
                  plotsize=plotsize)

  savefig(plt, "forwardflux_fromsite1.png")

  data = [table[:, 1:p["number_of_spin_sites"]]
          for (p, table) in zip(parameter_lists, backwardflux_super)]
  plt = groupplot(timesteps_super,
                  data,
                  parameter_lists;
                  labels=reduce(hcat, ["k=$k" for k ∈ 1:max_n_spins]),
                  linestyles=reduce(hcat, repeat([:solid], max_n_spins)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"k \to 1",
                  plottitle="Flusso di spin all'indietro, dal primo sito",
                  plotsize=plotsize)

  savefig(plt, "backwardflux_fromsite1.png")

  # Grafico dell'occupazione degli autospazi di N della catena di spin
  # ------------------------------------------------------------------
  # L'ultimo valore di ciascuna riga rappresenta la somma di tutti i
  # restanti valori.
  N = size(chain_levels_super[begin], 2) - 1
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

  # Grafico dell'occupazione dei livelli degli oscillatori
  # ------------------------------------------------------
  for (list, pos) in zip([osc_levels_left_super, osc_levels_right_super],
                         ["sx", "dx"])
    N = size(list[begin], 2) - 1
    plt = groupplot(timesteps_super,
                    list,
                    parameter_lists;
                    labels=[string.(0:N-1)... "total"],
                    linestyles=[repeat([:solid], N)... :dash],
                    commonxlabel=L"\lambda\, t",
                    commonylabel=L"n",
                    plottitle="Occupazione degli autospazi "
                    * "dell'oscillatore $pos",
                    plotsize=plotsize)

    savefig(plt, "osc_levels_$pos.png")
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
