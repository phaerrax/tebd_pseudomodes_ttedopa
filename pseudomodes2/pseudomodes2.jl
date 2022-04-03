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
  current_allsites_super = []
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
  osc_dim_list = [p["oscillator_space_dimension"]
                  for p ∈ parameter_lists]
  kappa_list = [p["oscillator_spin_interaction_coefficient"]
                for p ∈ parameter_lists]
  if allequal(n_spin_sites_list) &&
    allequal(osc_dim_list) &&
    allequal(kappa_list)
    preload = true
    n_spin_sites = first(n_spin_sites_list)
    osc_dim = first(osc_dim_list)

    spin_range = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 2; dim=osc_dim)]

    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                   for n ∈ 1:length(sites)]

    # - la corrente tra siti
    current_allsites_ops = [current(sites, i, j)
                            for i ∈ spin_range
                            for j ∈ spin_range
                            if j > i]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")
  else
    preload = false
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
    ζ = parameters["oscillators_interaction_coefficient"]
    γ = parameters["oscillator_damping_coefficient"]
    ω₁ = parameters["oscillator1_frequency"]
    ω₂ = parameters["oscillator2_frequency"]
    T = parameters["temperature"]
    oscdim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    if !preload
      n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
      spin_range = 2 .+ (1:n_spin_sites)

      sites = [siteinds("HvOsc", 2; dim=osc_dim);
               siteinds("HvS=1/2", n_spin_sites);
               siteinds("HvOsc", 2; dim=osc_dim)]
    end

    # Definizione degli operatori nell'equazione di Lindblad
    # ======================================================
    localcfs = [ω₂; ω₁; repeat([ε], n_spin_sites); ω₁; ω₂]
    interactioncfs = [ζ; κ; repeat([1], n_spin_sites-1); κ; ζ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += γ * (op("Damping", sites[begin]; ω=ω₂, T=T) *
                         op("Id", sites[begin+1]))
    ℓlist[end] += γ * (op("Id", sites[end-1]) *
                       op("Damping", sites[end]; ω=ω₂, T=0))
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
      current_allsites_ops = [current(sites, i, j)
                              for i ∈ spin_range
                              for j ∈ spin_range
                              if j > i]

      # - la normalizzazione (cioè la traccia) della matrice densità
      full_trace = MPS(sites, "vecId")
    end

    current_adjsites_ops = [-2ζ*current(sites, 1, 2);
                            -2κ*current(sites, 2, 3);
                            [current(sites, j, j+1)
                             for j ∈ spin_range[1:end-1]];
                            -2κ*current(sites,
                                        spin_range[end],
                                        spin_range[end]+1);
                            -2ζ*current(sites,
                                        spin_range[end]+1,
                                        spin_range[end]+2)]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    ρ₀ = chain(parse_init_state_osc(sites[1],
                                    parameters["left_oscillator_initial_state"];
                                    ω=ω₂, T=T),
               parse_init_state_osc(sites[2],
                                    parameters["left_oscillator_initial_state"];
                                    ω=ω₁, T=T),
               parse_init_state(sites[spin_range],
                                parameters["chain_initial_state"]),
               parse_init_state_osc(sites[end-1], "empty"),
               parse_init_state_osc(sites[end], "empty"))

    # Osservabili
    # -----------
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) / trace(ρ) for N in num_op_list])
    current_adjsites(ρ) = real.([inner(j, ρ) / trace(ρ)
                                 for j ∈ current_adjsites_ops])
    function current_allsites(ρ)
      pairs = [(i,j) for i ∈ 1:n_spin_sites for j ∈ 1:n_spin_sites if j > i]
      mat = zeros(n_spin_sites, n_spin_sites)
      for (j, i) in zip(current_allsites_ops, pairs)
        mat[i...] = real(inner(j, ρ) / trace(ρ))
      end
      mat .-= transpose(mat)
      return Base.vec(mat')
    end

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout,
    normalisation,
    occnlist,
    current_allsites_list,
    current_adjsites_list,
    ranks =  evolve(ρ₀,
                    time_step_list,
                    parameters["skip_steps"],
                    parameters["TS_expansion_order"],
                    links_odd,
                    links_even,
                    parameters["MP_compression_error"],
                    parameters["MP_maximum_bond_dimension"];
                    fout=[trace,
                          occn,
                          current_allsites,
                          current_adjsites,
                          linkdims])

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    current_allsites_list = mapreduce(permutedims, vcat, current_allsites_list)
    current_adjsites_list = mapreduce(permutedims, vcat, current_adjsites_list)
    ranks = mapreduce(permutedims, vcat, ranks)
    

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    for (j, name) in enumerate([:occ_n_left;
                                [Symbol("occ_n_spin$n") for n = 1:n_spin_sites];
                                :occ_n_right])
      push!(dict, name => occnlist[:,j])
    end
    syms = [Symbol("current_$i/$j")
            for i ∈ 1:n_spin_sites
            for j ∈ 1:n_spin_sites]
    for (coln, s) ∈ enumerate(syms)
        push!(dict, s => current_allsites_list[:, coln])
    end
    len = n_spin_sites + 2
    for (j, name) in enumerate([Symbol("bond_dim$n")
                                for n ∈ 1:len-1])
      push!(dict, name => ranks[:,j])
    end
    push!(dict, :full_trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occ_n_super, occnlist)
    push!(current_allsites_super, current_allsites_list)
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
  plotsize = (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_super[begin], 2)
  plt = groupplot(timesteps_super,
                  occ_n_super,
                  parameter_lists;
                  labels=["L2" "L1" string.(1:N-2)... "R1" "R2"],
                  linestyles=[:dash :dash repeat([:solid], N-4)... :dash :dash],
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
  spinsonly = [mat[:, 3:end-2] for mat in occ_n_super]
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
  sums = [hcat(sum(mat[:, 1:2], dims=2),
               sum(mat[:, 3:end-2], dims=2),
               sum(mat[:, end-1:end], dims=2))
          for mat ∈ occ_n_super]
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
  sitelabels = ["L2"; "L1"; string.(1:N+1); "R1"; "R2"]
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

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end