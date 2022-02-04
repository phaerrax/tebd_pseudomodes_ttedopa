#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV
using DifferentialEquations

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
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/operators.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# Differisce da "chain_with_damped_oscillators" in quanto qui
# indago l'andamento di quantità diverse e lo confronto con una
# soluzione esatta (per un numero di spin sufficientemente basso).

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
  normalisation_super = []
  bond_dimensions_super = []
  numeric_occ_n_super = []
  numeric_tsteps_super = []

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
    n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
    spin_range = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=osc_dim)]

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
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             time_step,
                             parameters["TS_expansion_order"])

    # Osservabili da misurare
    # =======================
    # - i numeri di occupazione
    single_ex_states = [chain(MPS(sites[1:1], "vecId"),
                              single_ex_state(sites[2:end-1], k),
                              MPS(sites[end:end], "vecId"))
                        for k = 1:n_spin_sites]
    osc_num_sx = MPS(sites, ["vecN"; repeat(["vecId"], n_spin_sites+1)])
    osc_num_dx = MPS(sites, [repeat(["vecId"], n_spin_sites+1); "vecN"])

    occ_n_list = [osc_num_sx; single_ex_states; osc_num_dx]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Lo stato iniziale della catena è "spin_initial_state" per lo spin
    # attaccato all'oscillatore, "empty" per gli altri.
    if n_spin_sites >= 2
    current_state = chain(parse_init_state_osc(sites[1],
                                 parameters["left_oscillator_initial_state"];
                                 ω=ω, T=T),
                          parse_spin_state(sites[2],
                                           parameters["spin_initial_state"]),
                          parse_init_state(sites[3:end-1],
                                           "empty"),
                          parse_init_state_osc(sites[end], "empty"))
    else
    current_state = chain(parse_init_state_osc(sites[1],
                                 parameters["left_oscillator_initial_state"];
                                 ω=ω, T=T),
                          parse_spin_state(sites[2],
                                           parameters["spin_initial_state"]),
                          parse_init_state_osc(sites[end], "empty"))
    end

    full_trace = MPS(sites, "vecId")

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = Vector{Real}[chop.([inner(s, current_state) for s in occ_n_list])]
    bond_dimensions = Vector{Int}[linkdims(current_state)]
    normalisation = Real[real(inner(full_trace, current_state))]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state,
                            cutoff=max_err,
                            maxdim=max_dim)
      if skip_count % skip_steps == 0
        #=
        Calcolo dapprima la traccia della matrice densità. Se non devia
        eccessivamente da 1, in ogni caso influisce sul valore delle
        osservabili che calcolo successivamente, che si modificano dello
        stesso fattore, e devono essere quindi corrette di un fattore pari
        al reciproco della traccia.
        =#
        trace = real(inner(full_trace, current_state))

        push!(normalisation,
              trace)
        push!(occ_n,
              [real(inner(s, current_state)) for s in occ_n_list] ./ trace)
        push!(bond_dimensions,
              linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    # Ora calcolo la soluzione esatta, integrando l'equazione di Lindblad.
    # Parto con il primo spin su, gli altri giù, e gli oscillatori all'equilibrio
    # termico (alle rispettive temperature).
    if T == 0
      mat = zeros(Float64, osc_dim, osc_dim)
      mat[1, 1] = 1
    else
      mat = exp(-ω / T * num(osc_dim))
      mat /= tr(mat)
    end
    initstate = mat ⊗ σ⁺
    for j ∈ 2:n_spin_sites
      initstate = initstate ⊗ σ⁻
    end
    mat = zeros(Float64, osc_dim, osc_dim)
    mat[1, 1] = 1
    initstate = initstate ⊗ mat

    # Operatori numero, per calcolare i numeri di occupazione
    spin_num_list = [[i == j ? [1 0; 0 0] : I₂ for i ∈ 1:n_spin_sites]
                     for j ∈ 1:n_spin_sites]
    num_ops = [num(osc_dim) ⊗ Matrix{Float64}(I, 2^n_spin_sites * osc_dim,
                 2^n_spin_sites * osc_dim),
               [Matrix{Float64}(I, osc_dim, osc_dim) ⊗
                 reduce(⊗, list; init=[1]) ⊗
                 Matrix{Float64}(I, osc_dim, osc_dim)
                 for list ∈ spin_num_list]...,
               Matrix{Float64}(I, 2^n_spin_sites * osc_dim,
                 2^n_spin_sites * osc_dim) ⊗
                 num(osc_dim)]

    numeric_occ_n = Real[real(tr(N * initstate)) for N ∈ num_ops]

    HoscL = ω * num(osc_dim) ⊗
              Matrix{Float64}(I, 2*n_spin_sites * osc_dim,
              2*n_spin_sites * osc_dim)

    h1list = [[i == j ? σᶻ : I₂ for i ∈ 1:n_spin_sites]
              for j ∈ 1:n_spin_sites]
    h1 = [reduce(⊗, list; init=[1])
          for list ∈ h1list]
    h2list = [[i == j ? σ⁺⊗σ⁻+σ⁻⊗σ⁺ : I₂ for i ∈ 1:n_spin_sites-1]
              for j ∈ 1:n_spin_sites-1]
    h2 = [reduce(⊗, list; init=[1])
          for list ∈ h2list]
    Hspin = 0.5 * Matrix{Float64}(I, osc_dim, osc_dim) ⊗
              (ε * sum(h1; init=[0 0; 0 0]) - sum(h2; init=[0 0; 0 0])) ⊗
              Matrix{Float64}(I, osc_dim, osc_dim)

    HoscR = Matrix{Float64}(I, 2^n_spin_sites * osc_dim,
                            2^n_spin_sites * osc_dim) ⊗ (ω * num(osc_dim))

    X = a⁺(osc_dim) + a⁻(osc_dim)
    HintL = κ * X ⊗ σˣ ⊗ Matrix{Float64}(I, 2^(n_spin_sites-1) * osc_dim,
                                         2^(n_spin_sites-1) * osc_dim)
    HintR = κ * Matrix{Float64}(I, 2^(n_spin_sites-1) * osc_dim,
                                2^(n_spin_sites-1) * osc_dim) ⊗ σˣ ⊗ X

    H = HoscL + HintL + Hspin + HintR + HoscR

    n = (ℯ^(ω / T) - 1)^(-1)
    jumpoperators = [sqrt(γₗ * (n+1)) * a⁻(osc_dim) ⊗ Matrix{Float64}(I, 2^n_spin_sites * osc_dim, 2^n_spin_sites * osc_dim),
                     sqrt(γₗ * n) * a⁺(osc_dim) ⊗ Matrix{Float64}(I, 2^n_spin_sites * osc_dim, 2^n_spin_sites * osc_dim),
                     sqrt(γᵣ) * Matrix{Float64}(I, 2^n_spin_sites * osc_dim, 2^n_spin_sites * osc_dim) ⊗ a⁻(osc_dim)]

    function lindblad!(∂ₜρ, ρ, par, t)
      # ∂ₜρ = ℒ(ρ) = -i[H,ρ] + ∑ᵢ(JᵢρJᵢ' - ½ Jᵢ'Jᵢρ - ½ ρJᵢ'Jᵢ)
      ∂ₜρ = -im * (H*ρ - ρ*H) +
            sum([J*ρ*J' - 0.5J'*J*ρ - 0.5ρ*J'*J for J in jumpoperators])
    end
    problem = ODEProblem(lindblad!,
                         initstate,
                         (time_step_list[begin], time_step_list[end]))
    solution = solve(problem, saveat=time_step_list)

    numeric_occ_n = [[real(tr(N * ρₜ)) for N ∈ num_ops]
                     for (ρₜ,_) ∈ tuples(solution)]

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n = 1:n_spin_sites];
                              :occ_n_right])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(bond_dimensions...)
    len = n_spin_sites + 2
    for (j, name) in enumerate([Symbol("bond_dim$n")
                                for n ∈ 1:len-1])
      push!(dict, name => tmp_list[j,:])
    end
    push!(dict, :full_trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list[1:skip_steps:end])
    push!(occ_n_super, permutedims(hcat(occ_n...)))
    push!(numeric_occ_n_super, permutedims(hcat(numeric_occ_n...)))
    push!(numeric_tsteps_super, [t for (_,t) ∈ tuples(solution)])
    push!(bond_dimensions_super, permutedims(hcat(bond_dimensions...)))
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

  N = size(occ_n_super[begin])[2]
  plt = groupplot(numeric_tsteps_super,
                  numeric_occ_n_super,
                  parameter_lists;
                  labels=["L" string.(1:N-2)... "R"],
                  linestyles=[:dash repeat([:solid], N-2)... :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (sol. numerica)",
                  plotsize=plotsize)

  savefig(plt, "occ_n_numeric.png")

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

  ## Grafico della corrente di spin
  ## ------------------------------
  #N = size(spin_current_super[begin])[2]
  #plt = groupplot(timesteps_super,
  #                spin_current_super,
  #                parameter_lists;
  #                labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
  #                linestyles=hcat(repeat([:solid], N)...),
  #                commonxlabel=L"\lambda\, t",
  #                commonylabel=L"j_{k,k+1}(t)",
  #                plottitle="Corrente di spin",
  #                plotsize=plotsize)

  #savefig(plt, "spin_current.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
