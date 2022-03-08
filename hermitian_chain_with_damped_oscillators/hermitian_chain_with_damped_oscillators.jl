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
  nearcurrent_super = []
  farcurrent_super = []
  osccurrent_super = []
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
    κ = first(kappa_list)

    spin_range = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=osc_dim)]

    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                      [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                  for n ∈ 1:length(sites)]

    # - la corrente tra siti
    nearcurrentops = [κ*current(sites, 1, 2);
                      [-0.5*current(sites, j, j+1)
                       for j ∈ spin_range[1:end-1]];
                      κ*current(sites, spin_range[end], spin_range[end]+1)]
    farcurrentops = [current(sites, 2, j) for j ∈ spin_range[2:end]]
    osccurrentop = current(sites, 1, eachindex(sites)[end])

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
                             for n=0:osc_dim-1]
    osc_levels_projs_right = [embed_slice(sites,
                                          n_spin_sites+2:n_spin_sites+2,
                                          osc_levels_proj(sites[end], n))
                              for n=0:osc_dim-1]

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
    γₗ = parameters["oscillator_damping_coefficient_left"]
    γᵣ = parameters["oscillator_damping_coefficient_right"]
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    if !preload
      n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
      spin_range = 1 .+ (1:n_spin_sites)

      sites = [siteinds("HvOsc", 1; dim=osc_dim);
               siteinds("HvS=1/2", n_spin_sites);
               siteinds("HvOsc", 1; dim=osc_dim)]
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
    if !preload
      # - i numeri di occupazione
      num_op_list = [MPS(sites,
                        [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                    for n ∈ 1:length(sites)]

      # - la corrente tra siti
      nearcurrentops = [κ*current(sites, 1, 2);
                        [-0.5*current(sites, j, j+1)
                        for j ∈ spin_range[1:end-1]];
                        κ*current(sites, spin_range[end], spin_range[end]+1)]
      farcurrentops = [-0.5*current(sites, 2, j)
                       for j ∈ spin_range[2:end]]
      osccurrentop = current(sites, 1, eachindex(sites)[end])

      # - l'occupazione degli autospazi dell'operatore numero
      num_eigenspace_projs = [embed_slice(sites,
                                          spin_range,
                                          level_subspace_proj(sites[spin_range], n))
                              for n=0:n_spin_sites]

      # - l'occupazione dei livelli degli oscillatori
      osc_levels_projs_left = [embed_slice(sites,
                                           1:1,
                                           osc_levels_proj(sites[1], n))
                               for n=0:osc_dim-1]
      osc_levels_projs_right = [embed_slice(sites,
                                            n_spin_sites+2:n_spin_sites+2,
                                            osc_levels_proj(sites[end], n))
                                for n=0:osc_dim-1]

      # - la normalizzazione (cioè la traccia) della matrice densità
      full_trace = MPS(sites, "vecId")
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
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
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) / trace(ρ) for N in num_op_list])
    nearcurrent(ρ) = real.([inner(j, ρ) / trace(ρ) for j in nearcurrentops])
    farcurrent(ρ) = real.([inner(j, ρ) / trace(ρ) for j in farcurrentops])
    osccurrent(ρ) = real(inner(osccurrentop, ρ) / trace(ρ))
    chainlevels(ρ) = real.(levels(num_eigenspace_projs, trace(ρ)^(-1) * ρ))
    osclevelsL(ρ) = real.(levels(osc_levels_projs_left, trace(ρ)^(-1) * ρ))
    osclevelsR(ρ) = real.(levels(osc_levels_projs_right, trace(ρ)^(-1) * ρ))

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout,
    normalisation,
    occnlist,
    nearcurrentlist,
    farcurrentlist,
    osccurrentlist,
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
                                   nearcurrent,
                                   farcurrent,
                                   osccurrent,
                                   linkdims,
                                   osclevelsL,
                                   osclevelsR,
                                   chainlevels])

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    nearcurrentlist = mapreduce(permutedims, vcat, nearcurrentlist)
    farcurrentlist = mapreduce(permutedims, vcat, farcurrentlist)
    ranks = mapreduce(permutedims, vcat, ranks)
    chainlevelslist = mapreduce(permutedims, vcat, chainlevelslist)
    osclevelsLlist = mapreduce(permutedims, vcat, osclevelsLlist)
    osclevelsRlist = mapreduce(permutedims, vcat, osclevelsRlist)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    for (j, name) in enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n = 1:n_spin_sites];
                              :occ_n_right])
      push!(dict, name => occnlist[:,j])
    end
    for (j, name) in enumerate([Symbol("near_current$n")
                                for n ∈ 1:size(nearcurrentlist, 2)])
      push!(dict, name => nearcurrentlist[:,j])
    end
    for (j, name) in enumerate([Symbol("far_current$n")
                                for n ∈ 1:size(farcurrentlist, 2)])
      push!(dict, name => farcurrentlist[:,j])
    end
    push!(dict, :osccurrent => osccurrentlist)
    for (j, name) in enumerate([Symbol("levels_left$n") for n = 0:osc_dim-1])
      push!(dict, name => osclevelsLlist[:,j])
    end
    for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
      push!(dict, name => chainlevelslist[:,j])
    end
    for (j, name) in enumerate([Symbol("levels_right$n") for n = 0:osc_dim:-1])
      push!(dict, name => osclevelsRlist[:,j])
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
    push!(nearcurrent_super, nearcurrentlist)
    push!(farcurrent_super, farcurrentlist)
    push!(osccurrent_super, osccurrentlist)
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
  plotsize = (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

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
  N = size(nearcurrent_super[begin], 2)
  sitelabels = ["L"; string.(1:N+1); "R"]
  plt = groupplot(timesteps_super,
                  nearcurrent_super,
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

  N = size(farcurrent_super[begin], 2)
  plt = groupplot(timesteps_super,
                  farcurrent_super,
                  parameter_lists;
                  labels=hcat(["(1,$(j+1))" for j ∈ 1:N]...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle j_{1,k}\rangle",
                  plottitle="Corrente di spin (dal primo spin)",
                  plotsize=plotsize)

  savefig(plt, "spin_current_from_first_spin.png")

  # Grafico della corrente tra gli oscillatori
  # ------------------------------------------
  plt = unifiedplot(timesteps_super,
                    osccurrent_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"\lambda\, t",
                    ylabel=L"\langle j_{L,R}\rangle",
                    plottitle="Corrente tra gli oscillatori",
                    plotsize=plotsize)

  savefig(plt, "current_btw_oscillators.png")

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

  # Grafico dell'occupazione dei livelli degli oscillatori
  # ------------------------------------------------------
  for (list, pos) in zip([osc_levels_left_super, osc_levels_right_super],
                         ["sx", "dx"])
    N = size(list[begin])[2] - 1
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
