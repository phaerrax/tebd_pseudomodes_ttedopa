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

let
  parameter_lists = load_parameters(ARGS)
  tot_sim_n = length(parameter_lists)

  # Se il primo argomento da riga di comando è una cartella (che 
  # dovrebbe contenere i file dei parametri), mi sposto subito in tale
  # posizione in modo che i file di output, come grafici e tabelle,
  # siano salvati insieme ai file di parametri.
  prev_dir = pwd()
  if isdir(ARGS[1])
    cd(ARGS[1])
  end

  # Le seguenti liste conterranno i risultati della simulazione
  # per ciascuna lista di parametri fornita.
  timesteps_super = []
  occn_super = []
  spin_current_super = []
  bond_dimensions_super = []
  spin_chain_levels_super = []
  range_osc_left_super = []
  range_spins_super = []
  range_osc_right_super = []
  osc_chain_coefficients_left_super = []
  osc_chain_coefficients_right_super = []
  snapshot_super = []
  normalisation_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================
    # Numero di siti e dimensioni
    n_spin_sites = parameters["number_of_spin_sites"]

    n_osc_left = parameters["left_bath"]["number_of_oscillators"]
    max_osc_dim_left = parameters["left_bath"]["maximum_oscillator_space_dimension"]
    osc_dims_decay_left = parameters["left_bath"]["oscillator_space_dimensions_decay"]

    n_osc_right = parameters["right_bath"]["number_of_oscillators"]
    max_osc_dim_right = parameters["right_bath"]["maximum_oscillator_space_dimension"]
    osc_dims_decay_right = parameters["right_bath"]["oscillator_space_dimensions_decay"]

    sites = [
             reverse([siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_left, max_osc_dim_left, osc_dims_decay_left)]);
             repeat([siteind("S=1/2")], n_spin_sites);
             [siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_right, max_osc_dim_right, osc_dims_decay_right)]
            ]
    for n ∈ eachindex(sites)
      sites[n] = addtags(sites[n], "n=$n")
    end

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    # Parametri tecnici per la simulazione
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # Istanti di tempo da attraversare
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Parametri della catena di spin
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    
    # Parametri delle catene di oscillatori
    # Ci sono due casi qui, in base a cosa viene specificato nel JSON:
    # 1) viene indicato un file alla voce "X_chain_coefficients_file"
    #    dove X è "left" o "right". In questo caso, prendo i
    #    coefficienti da quel file.
    # 2) viene descritta una densità spettrale, con i suoi parametri,
    #    alle voci "spectral_density_parameters" e "…_function". In
    #    questo caso, calcolo i coefficienti con le funzioni del
    #    T-TEDOPA come sempre.
    # In ogni caso, la temperatura dell'ambiente è completamente
    # inserita nei coefficienti, così come la frequenza di taglio,
    # quindi non verranno specificati.
    # È ammessa pure una situazione mista, in cui una catena è data
    # tramite i coefficienti e l'altra tramite una funzione.
    # Ciò che non è ammesso è dare contemporaneamente funzione e
    # coefficienti per il medesimo ambiente: qui lancio un errore.
    leftp = parameters["left_bath"]
    if (haskey(leftp, "chain_coefficients_file") &&
        !haskey(leftp, "spectral_density_function"))
      ccoeffs = DataFrame(CSV.File(leftp["chain_coefficients_file"]))
      if size(ccoeffs, 1) ≥ n_osc_left
        # Se è >, allora i coefficienti in eccesso vengono ignorati;
        # se è =, allora tutto combacia e tutti i coefficienti sono
        # considerati.
        Ωₗ = ccoeffs[1:n_osc_left, :loc]
        ηₗ = ccoeffs[1,            :int]
        κₗ = ccoeffs[2:n_osc_left, :int]
      else
        error("File $(leftp["chain_coefficients_file"]) does "*
              "not contain enough coefficients.")
      end
    elseif (!haskey(leftp, "chain_coefficients_file") &&
            haskey(leftp, "spectral_density_function"))
      fn = leftp["spectral_density_function"]
      tmp = eval(Meta.parse("(a, x) -> " * fn))
      sdf = x -> tmp(leftp["spectral_density_parameters"], x)

      ωc = leftp["frequency_cutoff"]
      T = leftp["temperature"]

      if T == 0
        (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(sdf,
                                            (0, ωc),
                                            n_osc_left-1;
                                            Nquad=leftp["PolyChaos_nquad"],
                                            discretization=lanczos)
      else
        sdf_thermalised = ω -> thermalisedJ(sdf, ω, T)
        (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(sdf_thermalised,
                                            (-ωc, 0, ωc),
                                            n_osc_left-1;
                                            Nquad=leftp["PolyChaos_nquad"],
                                            discretization=lanczos)
      end
    end

    # Ora lo stesso, ma con l'ambiente di destra...
    rightp = parameters["right_bath"]
    if (haskey(rightp, "chain_coefficients_file") &&
        !haskey(rightp, "spectral_density_function"))
      ccoeffs = DataFrame(CSV.File(rightp["chain_coefficients_file"]))
      if size(ccoeffs, 1) ≥ n_osc_right
        # Se è >, allora i coefficienti in eccesso vengono ignorati;
        # se è =, allora tutto combacia e tutti i coefficienti sono
        # considerati.
        Ωᵣ = ccoeffs[1:n_osc_right, :loc]
        ηᵣ = ccoeffs[1,             :int]
        κᵣ = ccoeffs[2:n_osc_right, :int]
      else
        error("File $(rightp["chain_coefficients_file"]) does "*
              "not contain enough coefficients.")
      end
    elseif (!haskey(rightp, "chain_coefficients_file") &&
            haskey(rightp, "spectral_density_function"))
      fn = rightp["spectral_density_function"]
      tmp = eval(Meta.parse("(a, x) -> " * fn))
      sdf = x -> tmp(rightp["spectral_density_parameters"], x)

      ωc = rightp["frequency_cutoff"]
      T = rightp["temperature"]

      if T == 0
        (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(sdf,
                                            (0, ωc),
                                            n_osc_right-1;
                                            Nquad=rightp["PolyChaos_nquad"],
                                            discretization=lanczos)
      else
        sdf_thermalised = ω -> thermalisedJ(sdf, ω, T)
        (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(sdf_thermalised,
                                            (-ωc, 0, ωc),
                                            n_osc_right-1;
                                            Nquad=rightp["PolyChaos_nquad"],
                                            discretization=lanczos)
      end
    end

    # Raccolgo i coefficienti in due array (uno per quelli a sx, l'altro per
    # quelli a dx) per poterli disegnare assieme nei grafici.
    # (I coefficienti κ sono uno in meno degli Ω! Pareggio le lunghezze
    # inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ωₗ [0; κₗ]]
    osc_chain_coefficients_right = [Ωᵣ [0; κᵣ]]
    # Adesso che ho i coefficienti, posso anche dimenticarmi di tutti i
    # parametri usati per costruire gli ambienti.

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#

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

    # Osservabili da misurare
    # =======================
    # - la corrente di spin
    # (Ci metto anche quella tra spin e primi oscillatori)
    spin_current_ops = [-2ηₗ*current(sites,
                                     range_spins[1]-1,
                                     range_spins[1]);
                        [current(sites, j, j+1)
                         for j ∈ range_spins[1:end-1]];
                        -2ηᵣ*current(sites,
                                     range_spins[end],
                                     range_spins[end]+1)]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto; mi riservo di decidere
    # volta per volta come inizializzare l'oscillatore più a destra nella
    # catena a sinistra, per motivi di diagnostica.
    osc_sx_init_state = MPS(sites[range_osc_left], "0")
    spin_init_state = parse_init_state(sites[range_spins],
                                       parameters["chain_initial_state"])
    osc_dx_init_state = MPS(sites[range_osc_right], "0")
    ψ₀ = chain(osc_sx_init_state, spin_init_state, osc_dx_init_state)

    # Osservabili da misurare
    # -----------------------
    occn(ψ) = real.(expect(ψ, "N")) ./ norm(ψ)^2
    spincurrent(ψ) = real.([inner(ψ', j, ψ) for j ∈ spin_current_ops]) ./ norm(ψ)^2
    # Calcolo i ranghi tra tutti gli spin, più quelli tra gli spin e
    # i primi oscillatori, quelli appena attaccati alla catena.
    # Questo risolve anche il problema di come trattare questa funzione
    # quando c'è un solo spin nella catena.
    spinlinkdims(ψ) = [linkdims(ψ)[range_osc_left[end] : range_spins[end]];
                       maxlinkdim(ψ)]

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout, normalisation, occnlist, spincurrentlist, ranks = evolve(ψ₀,
                             time_step_list,
                             parameters["skip_steps"],
                             parameters["TS_expansion_order"],
                             links_odd,
                             links_even,
                             parameters["MP_compression_error"],
                             parameters["MP_maximum_bond_dimension"];
                             fout=[norm, occn, spincurrent, spinlinkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    spincurrentlist = mapreduce(permutedims, vcat, spincurrentlist)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    sitelabels = [string.("L", n_osc_left:-1:1);
                  string.("S", 1:n_spin_sites);
                  string.("R", 1:n_osc_right)]
    for (n, label) ∈ enumerate(sitelabels)
      push!(dict, Symbol(string("occn_", label)) => occnlist[:, n])
    end
    for n ∈ -1:n_spin_sites-1
      from = sitelabels[range_spins[begin] + n]
      to   = sitelabels[range_spins[begin] + n+1]
      sym  = "current_adjsites_$from/$to"
      push!(dict, Symbol(sym) => spincurrentlist[:, n+2])
      sym  = "rank_$from/$to"
      push!(dict, Symbol(sym) => ranks[:, n+2])
    end
    push!(dict, :maxrank => ranks[:, end])
    push!(dict, :norm => normalisation)

    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occn_super, occnlist)
    push!(spin_current_super, spincurrentlist)
    push!(bond_dimensions_super, ranks)
    push!(range_osc_left_super, range_osc_left)
    push!(range_spins_super, range_spins)
    push!(range_osc_right_super, range_osc_right)
    push!(osc_chain_coefficients_left_super, osc_chain_coefficients_left)
    push!(osc_chain_coefficients_right_super, osc_chain_coefficients_right)
    push!(snapshot_super, occnlist[end,:])
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

  # Grafico dei numeri di occupazione (oscillatori sx)
  # --------------------------------------------------
  plt = groupplot(timesteps_super,
                  [mat[:, range]
                   for (mat, range) ∈ zip(occn_super, range_osc_left_super)],
                  parameter_lists;
                  rescale=false,
                  labels=[reduce(hcat, string.("l", reverse(eachindex(range))))
                          for range ∈ range_osc_left_super], # ["l1" ... "lN"]
                  linestyles=[reduce(hcat, repeat([:solid], length(range)))
                              for range ∈ range_osc_left_super],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione "*
                             "(oscillatori a sx)",
                  plotsize=plotsize,
                 filenameastitle=true)

  savefig(plt, "occn_osc_left.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  plt = groupplot(timesteps_super,
                  [mat[:, range]
                   for (mat, range) ∈ zip(occn_super, range_spins_super)],
                  parameter_lists;
                  rescale=false,
                  labels=[reduce(hcat, string.("s", eachindex(range)))
                          for range ∈ range_spins_super], # ["s1" ... "sN"]
                  linestyles=[reduce(hcat, repeat([:solid], length(range)))
                              for range ∈ range_spins_super],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (spin)",
                  plotsize=plotsize,
                 filenameastitle=true)

  savefig(plt, "occn_spins.png")
  
  # Grafico dei numeri di occupazione (oscillatori dx)
  # --------------------------------------------------
  plt = groupplot(timesteps_super,
                  [mat[:, range]
                   for (mat, range) ∈ zip(occn_super, range_osc_right_super)],
                  parameter_lists;
                  rescale=false,
                  labels=[reduce(hcat, string.("s", eachindex(range)))
                          for range ∈ range_osc_right_super], # ["s1" ... "sN"]
                  linestyles=[reduce(hcat, repeat([:solid], length(range)))
                              for range ∈ range_osc_right_super],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (oscillatori dx)",
                  plotsize=plotsize,
                 filenameastitle=true)

  savefig(plt, "occn_osc_right.png")

  # Grafico dei numeri di occupazione (tot oscillatori + tot catena)
  # ----------------------------------------------------------------
  sums = [[sum(mat[:, rangeL], dims=2);;
           sum(mat[:, rangeS], dims=2);;
           sum(mat[:, rangeR], dims=2);;
           sum(mat, dims=2)]
          for (mat, rangeL, rangeS, rangeR) ∈ zip(occn_super,
                                                  range_osc_left_super,
                                                  range_spins_super,
                                                  range_osc_right_super)]
  plt = groupplot(timesteps_super,
                  sums,
                  parameter_lists;
                  labels=["osc. sx" "catena" "osc. dx" "tutti"],
                  linestyles=[:solid :solid :solid :dash],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione (sommati)",
                  plotsize=plotsize,
                 filenameastitle=true)
                  
  savefig(plt, "occn_sums.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  ranklabels=[reduce(hcat, ["(l1,s1)";
                            ["(s$j,s$(j+1))" for j ∈ 1:size(v, 2)-3];
                            "(s10,r1)";
                            "max"])
                          for v ∈ bond_dimensions_super]
  ranklinestyles = [reduce(hcat, [repeat([:solid], size(v, 2)-1);
                                  :dash])
                    for v ∈ bond_dimensions_super]

  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=ranklabels,
                  linestyles=ranklinestyles,
                  commonxlabel=L"t",
                  commonylabel=L"\chi_{k,k+1}(t)",
                  plottitle="Ranghi del MPS",
                  plotsize=plotsize,
                 filenameastitle=true)

  savefig(plt, "bond_dimensions.png")

  # Grafico della corrente di spin
  # ------------------------------
  plt = groupplot(timesteps_super,
                  spin_current_super,
                  parameter_lists;
                  labels=[reduce(hcat,
                                 ["($(j-1),$j)" for j ∈ 1:size(c, 2)])
                          for c in spin_current_super],
                  linestyles=[reduce(hcat,
                                     repeat([:solid], size(c, 2)))
                              for c in spin_current_super],
                  commonxlabel=L"t",
                  commonylabel=L"j_{k,k+1}(t)",
                  plottitle="Corrente di spin",
                  plotsize=plotsize,
                 filenameastitle=true)

  savefig(plt, "spin_current.png")

  # Grafico dei coefficienti della chain map
  # ----------------------------------------
  plt = groupplot([1:p["left_bath"]["number_of_oscillators"]
                   for p ∈ parameter_lists],
                  osc_chain_coefficients_left_super,
                  parameter_lists;
                  labels=["Ωᵢ" "κᵢ"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                             "oscillatori (sx)",
                  plotsize=plotsize,
                 filenameastitle=true)

  savefig(plt, "osc_left_coefficients.png")

  plt = groupplot([1:p["right_bath"]["number_of_oscillators"]
                   for p ∈ parameter_lists],
                  osc_chain_coefficients_right_super,
                  parameter_lists;
                  labels=["Ωᵢ" "κᵢ"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                            "oscillatori (dx)",
                  plotsize=plotsize,
                 filenameastitle=true)

  savefig(plt, "osc_right_coefficients.png")

  # Istantanea dei numeri di occupazione alla fine
  # ----------------------------------------------
  plt = unifiedplot([[rangeL; rangeS; rangeR] for (rangeL, rangeS, rangeR)
                   ∈ zip(range_osc_left_super,
                         range_spins_super,
                         range_osc_right_super)],
                    snapshot_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"i",
                    ylabel="Numero di occupazione",
                    plottitle="Numeri di occupazione, alla fine",
                    plotsize=plotsize,
                   filenameastitle=true)

  savefig(plt, "snapshot.png")

  plt = unifiedplot(range_spins_super,
                    [occn[range_spins] for (occn, range_spins)
                     ∈ zip(snapshot_super, range_spins_super)],
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"i",
                    ylabel="Numero di occupazione",
                    plottitle="Numeri di occupazione, alla fine, solo spin",
                    plotsize=plotsize,
                   filenameastitle=true)

  savefig(plt, "snapshot_spins.png")

  # Grafico della norma dello stato
  # -------------------------------
  # Questo serve per controllare che rimanga sempre pari a 1.
  plt = unifiedplot(timesteps_super,
                    normalisation_super,
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"t",
                    ylabel=L"\Vert\psi(t)\Vert",
                    plottitle="Norma dello stato",
                    plotsize=plotsize,
                   filenameastitle=true)

  savefig(plt, "normalisation.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
