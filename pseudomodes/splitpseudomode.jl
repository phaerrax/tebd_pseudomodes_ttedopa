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
  current_allsites_super = []
  current_adjsites_super = []
  bond_dimensions_super = []
  chain_levels_super = []
  osc_levels_left_super = []
  osc_levels_right_super = []
  normalisation_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    @info "($current_sim_n di $tot_sim_n) Costruzione degli operatori di evoluzione temporale."

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
    Ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]

    n(T,ω) = T == 0 ? 0.0 : (ℯ^(ω/T)-1)^(-1)

    # Set new initial oscillator states as empty.
    parameters["left_oscillator_initial_state"] = "empty"

    # ITensors internal parameters
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # Transform the pseudomode into two zero-temperature new modes.
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ₁ = κ * sqrt( 1+n(T,Ω) )
    ω₁ = Ω
    γ₁ = γₗ
    #
    κ₂ = κ * sqrt( n(T,Ω) )
    ω₂ = -Ω
    γ₂ = γₗ
    #
    κᵣ = κ
    ωᵣ = Ω
    #γᵣ = γᵣ
    #
    η = 0
    T = 0
    oscdim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    range_spins = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=oscdim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=oscdim)]

    # Definizione degli operatori nell'equazione di Lindblad
    # ======================================================
    # Calcolo i coefficienti dell'Hamiltoniano trasformato.
    ω̃₁ = (ω₁*κ₁^2 + ω₂*κ₂^2 + 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)
    ω̃₂ = (ω₂*κ₁^2 + ω₁*κ₂^2 - 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)
    κ̃₁ = sqrt(κ₁^2 + κ₂^2)
    κ̃₂ = ((ω₂-ω₁)*κ₁*κ₂ + η*(κ₁^2 - κ₂^2)) / (κ₁^2 + κ₂^2)
    # Dato che T = 0 dappertutto, posso semplificare qualcosa.
    #γ̃₁⁺ = (κ₁^2*γ₁*(n(T,ω₁)+1) + κ₂^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    #γ̃₁⁻ = (κ₁^2*γ₁* n(T,ω₁)    + κ₂^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)
    #γ̃₂⁺ = (κ₂^2*γ₁*(n(T,ω₁)+1) + κ₁^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    #γ̃₂⁻ = (κ₂^2*γ₁* n(T,ω₁)    + κ₁^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)
    #γ̃₁₂⁺ = κ₁*κ₂*( γ₂*(n(T,ω₂)+1) - γ₁*(n(T,ω₁)+1) )  / (κ₁^2 + κ₂^2)
    #γ̃₁₂⁻ = κ₁*κ₂*( γ₂*n(T,ω₂)     - γ₁*n(T,ω₁) )      / (κ₁^2 + κ₂^2)
    γ̃₁⁺ = γ₁ # == γ₂
    γ̃₁⁻ = 0
    γ̃₂⁺ = γ₂
    γ̃₂⁻ = 0
    γ̃₁₂⁺ = 0 # perché γ₁ = γ₂
    γ̃₁₂⁻ = 0

    localcfs = [ω̃₂; ω̃₁; repeat([ε], n_spin_sites); ωᵣ]
    interactioncfs = [κ̃₂; κ̃₁; repeat([1], n_spin_sites-1); κᵣ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli operatori già creati gli operatori di dissipazione:
    # rimuovo direttamente quelli nulli.
    # · per il primo oscillatore a sinistra,
    ℓlist[1] += γ̃₂⁺ * op("Lindb+", sites[1]) * op("Id", sites[2])
    #ℓlist[1] += γ̃₂⁻ * op("Lindb-", sites[1]) * op("Id", sites[2])
    # · per il secondo oscillatore (occhio che non essendo più all'estremo
    #   della catena questo operatore viene diviso tra ℓ₁,₂ e ℓ₂,₃),
    ℓlist[1] += 0.5γ̃₁⁺ * op("Id", sites[1]) * op("Lindb+", sites[2])
    #ℓlist[1] += 0.5γ̃₁⁻ * op("Id", sites[1]) * op("Lindb-", sites[2])
    ℓlist[2] += 0.5γ̃₁⁺ * op("Lindb+", sites[2]) * op("Id", sites[3])
    #ℓlist[2] += 0.5γ̃₁⁻ * op("Lindb-", sites[2]) * op("Id", sites[3])
    # · l'operatore misto su (1) e (2),
    #ℓlist[1] += γ̃₁₂⁺ * mixedlindbladplus(sites[1], sites[2])
    #ℓlist[1] += γ̃₁₂⁻ * mixedlindbladminus(sites[1], sites[2])
    # · infine per l'oscillatore a destra, come al solito,
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
                        op("Damping", sites[end]; ω=ωᵣ, T=0))
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # Osservabili da misurare
    # =======================
    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                   for n ∈ 1:length(sites)]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")

    current_adjsites_ops = [current(sites, j, j+1)
                            for j ∈ range_spins[1:end-1]]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    @info "($current_sim_n di $tot_sim_n) Creazione dello stato iniziale."
    if parameters["left_oscillator_initial_state"] == "thermal"
      # L'ambiente sx è in equilibrio termico, quello dx è vuoto.
      # Lo stato iniziale della catena è dato da "chain_initial_state".
      # Per calcolare lo stato iniziale dei due oscillatori a sinistra:
      # 1) Calcolo la matrice densità dello stato termico
      HoscL = (ω̃₁ * num(oscdim) ⊗ id(oscdim) +
               ω̃₂ * id(oscdim) ⊗ num(oscdim) +
               κ̃₂ * (a⁺(oscdim) ⊗ a⁻(oscdim) +
                     a⁻(oscdim) ⊗ a⁺(oscdim)))
      M = exp(-1/T * HoscL)
      M /= tr(M)
      # 2) la vettorizzo sul prodotto delle basi hermitiane dei due siti
      v = vec(M, [êᵢ ⊗ êⱼ for (êᵢ, êⱼ) ∈ [Base.product(gellmannbasis(oscdim), gellmannbasis(oscdim))...]])
      # 3) inserisco il vettore in un tensore con gli Index degli oscillatori
      iv = itensor(v, sites[1], sites[2])
      # 4) lo decompongo in due pezzi con una SVD
      f1, f2, _, _ = factorize(iv, sites[1]; which_decomp="svd")
      # 5) rinomino il Link tra i due fattori come "Link,l=1" anziché
      #    "Link,fact" che è il Tag assegnato da `factorize`
      replacetags!(f1, "fact" => "l=1")
      replacetags!(f2, "fact" => "l=1")

      ρ₀ = chain(MPS([f1, f2]),
                 parse_init_state(sites[range_spins],
                                  parameters["chain_initial_state"]),
                 parse_init_state_osc(sites[end], "empty"))
    elseif parameters["left_oscillator_initial_state"] == "empty"
      ρ₀ = chain(parse_init_state_osc(sites[1], "empty"),
                 parse_init_state_osc(sites[2], "empty"),
                 parse_init_state(sites[range_spins],
                                  parameters["chain_initial_state"]),
                 parse_init_state_osc(sites[end], "empty"))
    end

    # Osservabili
    # -----------
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)
    current_adjsites(ρ) = real.([inner(j, ρ)
                                 for j ∈ current_adjsites_ops]) ./ trace(ρ)

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout,
      normalisation,
      occnlist,
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
                            current_adjsites,
                            linkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    current_adjsites_list = mapreduce(permutedims, vcat, current_adjsites_list)
    ranks = mapreduce(permutedims, vcat, ranks)

    @info "($current_sim_n di $tot_sim_n) Creazione delle tabelle di output."
    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    for (j, name) in enumerate([:occn_l2;
                                :occn_l1;
                                [Symbol("occn_s$n") for n = 1:n_spin_sites];
                                :occn_r1])
      push!(dict, name => occnlist[:,j])
    end
    for coln ∈ eachindex(current_adjsites_ops)
      s = Symbol("current_$coln/$(coln+1)")
      push!(dict, s => current_adjsites_list[:, coln])
    end
    for j ∈ 1:size(ranks, 2)
      s = Symbol("bond_dim$j")
      push!(dict, s => ranks[:,j])
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

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_super[begin], 2)
  plt = groupplot(timesteps_super,
                  occ_n_super,
                  parameter_lists;
                  labels=["L2" "L1" string.(1:N-3)... "R"],
                  linestyles=[:dash :dash repeat([:solid], N-3)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione",
                  plotsize=plotsize)

  savefig(plt, "occ_n_all.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  spinsonly = [mat[:, 3:end-1] for mat in occ_n_super]
  plt = groupplot(timesteps_super,
                  spinsonly,
                  parameter_lists;
                  labels=reduce(hcat, string.(1:N-3)),
                  linestyles=reduce(hcat, repeat([:solid], N-3)),
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
               sum(mat[:, 3:end-1], dims=2),
               mat[:, end])
          for mat ∈ occ_n_super]
  plt = groupplot(timesteps_super,
                  sums,
                  parameter_lists;
                  labels=["L1+L2" "spin" "R"],
                  linestyles=[:solid :dot :solid],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\sum_i\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (oscillatori + totale catena)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_sums.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  N = size(bond_dimensions_super[begin], 2)
  sitelabels = ["L2"; "L1"; string.(1:N-2); "R"]
  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=reduce(hcat,
                                ["($(sitelabels[j]),$(sitelabels[j+1]))"
                                 for j ∈ eachindex(sitelabels)[1:end-1]]),
                  linestyles=[:dash :dash repeat([:solid], N-3)... :dash],
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
  sitelabels = string.(1:N)
  plt = groupplot(timesteps_super,
                  current_adjsites_super,
                  parameter_lists;
                  labels=reduce(hcat,
                                ["($(sitelabels[j]),$(sitelabels[j+1]))"
                                 for j ∈ eachindex(sitelabels)[1:end-1]]),
                  linestyles=reduce(hcat, repeat([:solid], N)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle j_{k,k+1}\rangle",
                  plottitle="Corrente (tra siti adiacenti)",
                  plotsize=plotsize)

  savefig(plt, "current_btw_adjacent_sites.png")

  #max_n_spins = maximum([p["number_of_spin_sites"] for p in parameter_lists])
  #data = [table[:, 1:p["number_of_spin_sites"]]
  #        for (p, table) in zip(parameter_lists, current_allsites_super)]
  #plt = groupplot(timesteps_super,
  #                data,
  #                parameter_lists;
  #                labels=reduce(hcat, ["l=$l" for l ∈ 1:max_n_spins]),
  #                linestyles=reduce(hcat, repeat([:solid], max_n_spins)),
  #                commonxlabel=L"\lambda\, t",
  #                commonylabel=L"\langle j_{1,l}\rangle",
  #                plottitle="Corrente tra spin (dal 1° spin)",
  #                plotsize=plotsize)

  #savefig(plt, "spin_current_fromsite1.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
