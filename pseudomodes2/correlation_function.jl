#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV
using FFTW

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
  Xcorrelation_super = []
  FT_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    #κ = parameters["oscillator_spin_interaction_coefficient"]
    κ = 0.0
    ζ = parameters["oscillators_interaction_coefficient"]
    γ = parameters["oscillator_damping_coefficient"]
    ω₁ = parameters["oscillator1_frequency"]
    ω₂ = parameters["oscillator2_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
    spin_range = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 2; dim=osc_dim)]

    #= Definizione degli operatori nell'equazione di Lindblad
    ======================================================
    I siti del sistema sono numerati come segue:
    | 1 | 2 | ... | n_spin_sites | n_spin_sites+1 | n_spin_sites+2 |
      ↑   │                                │          ↑
      │   └───────────┬────────────────────┘          │
      │               │                               │
      │        catena di spin                         │
      oscillatore sx                               oscillatore dx
    =#
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

    # La correlazione
    # ===============
    # Detto X l'operatore dell'oscillatore "L1", la funzione di correlazione
    # da calcolare è
    # ⟨XₜX₀⟩ = tr(ρ₀XₜX₀) = vec(Xₜ)'vec(X₀ρ₀);
    # l'evoluzione di X avviene secondo l'equazione di Lindblad aggiunta,
    # che vettorizzata ha L' al posto di L:
    # vec(Xₜ) = exp(tL')vec(X₀).
    # Allora
    # ⟨XₜX₀⟩ = vec(X₀)'exp(tL')'vec(X₀ρ₀) = vec(X₀)'exp(tL)vec(X₀ρ₀).

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Lo stato iniziale qui è X₀ρ₀ (vedi eq. sopra).
    # In ρ₀, l'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    X₀ρ₀ = chain(MPS([state(sites[1], "ThermEq"; ω=ω₂, T=T)]),
                 MPS([state(sites[2], "X⋅Therm"; ω=ω₁, T=T)]),
                 parse_init_state(sites[spin_range],
                                  parameters["chain_initial_state"]),
                 MPS([state(sites[end-1], "0")]),
                 MPS([state(sites[end],   "0")]))

    # Osservabili
    # -----------
    X₀ = MPS(sites, [i == 2 ? "vecX" : "vecId" for i ∈ eachindex(sites)])
    correlation(ρ) = real(inner(X₀, ρ))

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout, correlationlist = evolve(X₀ρ₀,
                                   time_step_list,
                                   parameters["skip_steps"],
                                   parameters["TS_expansion_order"],
                                   links_odd,
                                   links_even,
                                   parameters["MP_compression_error"],
                                   parameters["MP_maximum_bond_dimension"];
                                   fout=[correlation])
    #FT = rfft(correlationlist) 
    ν(ω,T) = T == 0 ? 0.0 : (ℯ^(ω/T)-1)^(-1)
    c₀ = 2ν(ω₁,T) - 1
    c₀ = correlation(X₀ρ₀)
    expXcorrelation = [c₀ * ℯ^(-γ*t) * cos(ω₁*t) for t ∈ tout]
    Xcorrelationlist = hcat(correlationlist, expXcorrelation)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    push!(dict, :correlation_calc => Xcorrelationlist[:,1])
    push!(dict, :correlation_exp => Xcorrelationlist[:,2])
    #push!(dict, :spectraldensity => FT)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(Xcorrelation_super, Xcorrelationlist)
    push!(FT_super, correlationlist)
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

  # Grafico della funzione di correlazione
  # --------------------------------------
  plt = groupplot(timesteps_super,
                  Xcorrelation_super,
                  parameter_lists;
                  labels=["calculated" "expected"],
                  linestyles=[:solid :dash],
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle X(t)X(0)\rangle",
                  plottitle="Funzione di correlazione",
                  plotsize=plotsize)

  savefig(plt, "Xcorrelation.png")

  ## Grafico della (supposta) densità spettrale
  ## ------------------------------------------
  #plt = unifiedplot(timesteps_super,
  #                  FT_super,
  #                  parameter_lists;
  #                  linestyle=:solid,
  #                  xlabel=L"\omega",
  #                  ylabel=L"J(\omega)",
  #                  plottitle="Densità spettrale?",
  #                  plotsize=plotsize)

  #savefig(plt, "Xcorrelation.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
