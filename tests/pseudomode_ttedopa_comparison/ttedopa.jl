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
include(joinpath(libpath, "tedopa.jl"))

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
  occn_TTEDOPA_super = []
  range_spins_super = []
  range_osc_left_super = []
  normalisation_TTEDOPA_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    n_osc_left = parameters["number_of_oscillators_left"]
    n_osc_right = parameters["number_of_oscillators_right"]
    max_osc_dim = parameters["maximum_oscillator_space_dimension"]
    osc_dims_decay = parameters["oscillator_space_dimensions_decay"]
    sites = [
             reverse([siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_left, max_osc_dim, osc_dims_decay)]);
             repeat([siteind("S=1/2")], n_spin_sites);
             [siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_right, max_osc_dim, osc_dims_decay)]
            ]
    for n ∈ eachindex(sites)
      sites[n] = addtags(sites[n], "n=$n")
    end

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    # Impostazione dei parametri
    # ==========================

    # - parametri tecnici
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    Ω = parameters["spectral_density_peak"]
    γ = parameters["spectral_density_half_width"]
    κ = parameters["spectral_density_overall_factor"]
    T = parameters["temperature"]
    ωc = parameters["frequency_cutoff"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    J(ω) = κ^2/π * 0.5γ * (1 / ((0.5γ)^2 + (ω-Ω)^2) - 1 / ((0.5γ)^2 + (ω+Ω)^2))
    Jtherm = ω -> thermalisedJ(J, ω, T)
    Jzero  = ω -> thermalisedJ(J, ω, 0)
    (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(Jtherm,
                                        (-ωc, 0, ωc),
                                        n_osc_left-1;
                                        Nquad=nquad,
                                        discretization=lanczos)
    (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(Jzero,
                                        (0, ωc),
                                        n_osc_right-1;
                                        Nquad=nquad,
                                        discretization=lanczos)

    # Raccolgo i coefficienti in due array (uno per quelli a sx, l'altro per
    # quelli a dx) per poterli disegnare assieme nei grafici.
    # (I coefficienti κ sono uno in meno degli Ω! Per ora pareggio le lunghezze
    # inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ωₗ [0; κₗ]]
    osc_chain_coefficients_right = [Ωᵣ [0; κᵣ]]

    localcfs = [reverse(Ωₗ); repeat([ε], n_spin_sites); Ωᵣ]
    interactioncfs = [reverse(κₗ); ηₗ; repeat([1], n_spin_sites-1); ηᵣ; κᵣ]
    hlist = twositeoperators(sites, localcfs, interactioncfs)
    #
    function links_odd(τ)
      return [exp(-im * τ * h) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(-im * τ * h) for h in hlist[2:2:end]]
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto.
    ψ₀ = MPS(sites, [repeat(["0"], n_osc_left);
                     repeat(["Dn"], n_spin_sites);
                     repeat(["0"], n_osc_right)])

    # Osservabili da misurare
    # -----------------------
    occn(ψ) = real.(expect(ψ, "N")[1:range_spins[end]]) ./ norm(ψ)^2

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout, normalisation_TTEDOPA, occnlist = evolve(ψ₀,
                             time_step_list,
                             parameters["skip_steps"],
                             parameters["TS_expansion_order"],
                             links_odd,
                             links_even,
                             parameters["MP_compression_error"],
                             parameters["MP_maximum_bond_dimension"];
                             fout=[norm, occn])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occn_TTEDOPA = mapreduce(permutedims, vcat, occnlist)

    outfilename = replace(parameters["filename"], ".json" => ".dat")
    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time_TTEDOPA => tout)
    sitelabels = [string.("L", reverse(range_osc_left));
                  string.("S", eachindex(range_spins))]
    for (j, label) ∈ enumerate(sitelabels)
      push!(dict,
            Symbol(string("occn_TTEDOPA_", label)) => occn_TTEDOPA[:, j])
    end
    push!(dict, :norm_TTEDOPA => normalisation_TTEDOPA)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat~2")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occn_TTEDOPA_super, occn_TTEDOPA)
    push!(normalisation_TTEDOPA_super, normalisation_TTEDOPA)
    push!(range_spins_super, range_spins)
    push!(range_osc_left_super, range_osc_left)
  end

  # Grafici
  # =======
  plotsize = (600, 400)

  # Grafico della norma dello stato
  # -------------------------------
  plt = unifiedplot(timesteps_super,
                    normalisation_TTEDOPA_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\Vert\psi(t)\Vert",
                    plottitle="Norma dello stato (TTEDOPA)",
                    plotsize=plotsize)
  savefig(plt, "normalisation_TTEDOPA.png")

  # Grafico dei numeri di occupazione (solo oscillatori sx)
  # -------------------------------------------------------
  plt = groupplot(timesteps_super,
                  [occn[:, rn]
                   for (rn, occn) ∈ zip(range_osc_left_super, occn_TTEDOPA_super)],
                  parameter_lists;
                  labels=[reduce(hcat, ["L$n" for n ∈ reverse(rn)])
                          for rn ∈ range_osc_left_super],
                  linestyles=[reduce(hcat, repeat([:solid], length(rn)))
                              for rn ∈ range_osc_left_super],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione degli oscillatori sx, TTEDOPA",
                  plotsize=plotsize)
  savefig(plt, "occn_oscsx_TTEDOPA.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  plt = groupplot(timesteps_super,
                  [occn[:, rn]
                   for (rn, occn) ∈ zip(range_spins_super, occn_TTEDOPA_super)],
                  parameter_lists;
                  labels=[reduce(hcat, ["S$n" for n ∈ eachindex(rn)])
                          for rn ∈ range_spins_super],
                  linestyles=[reduce(hcat, repeat([:solid], length(rn)))
                              for rn ∈ range_spins_super],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione degli spin, TTEDOPA",
                  plotsize=plotsize)
  savefig(plt, "occn_spins_TTEDOPA.png")
  
  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
