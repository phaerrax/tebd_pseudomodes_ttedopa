#!/usr/bin/julia

using ITensors, DataFrames, LaTeXStrings, CSV, Plots
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

  # Precaricamento
  # ==============
  # Se in tutte le liste di parametri il numero di siti è lo stesso, posso
  # definire qui una volta per tutte alcuni elementi "pesanti" che servono dopo.
  S_sites_list        = [p["number_of_spin_sites"] for p in parameter_lists]
  max_osc_dim_list    = [p["maximum_oscillator_space_dimension"] for p in parameter_lists]
  osc_dims_decay_list = [p["oscillator_space_dimensions_decay"] for p in parameter_lists]
  L_sites_list        = [p["number_of_oscillators_left"] for p in parameter_lists]
  R_sites_list        = [p["number_of_oscillators_right"] for p in parameter_lists]
  if (allequal(S_sites_list) &&
      allequal(max_osc_dim_list) &&
      allequal(osc_dims_decay_list) &&
      allequal(L_sites_list) &&
      allequal(R_sites_list))
    preload = true
    n_spin_sites   = first(S_sites_list)
    max_osc_dim    = first(max_osc_dim_list)
    osc_dims_decay = first(osc_dims_decay_list)
    n_osc_left     = first(L_sites_list)
    n_osc_right    = first(R_sites_list)
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

    # - l'occupazione degli autospazi dell'operatore numero
    # Ad ogni istante proietto lo stato corrente sugli autostati
    # dell'operatore numero della catena di spin, vale a dire calcolo
    # (ψ,Pₙψ) dove ψ è lo stato corrente e Pₙ è il proiettore ortogonale
    # sull'n-esimo autospazio di N.
    projectors = [level_subspace_proj(sites[range_spins], n)
                  for n = 0:n_spin_sites]
    num_eigenspace_projs = [embed_slice(sites, range_spins, p)
                            for p in projectors]
  else
    preload = false
  end

  for (current_sim_n, parameters) in enumerate(parameter_lists)
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
    # La densità spettrale è data da
    # J(ω) = κ² ⋅ γ/π ⋅ 1/(γ² + (ω-Ω)²)
    T = parameters["temperature"]
    ωc = parameters["frequency_cutoff"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    if !preload
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
    end

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    J(ω) = κ^2 * 0.5γ/π * (1 / ((0.5γ)^2 + (ω-Ω)^2) - 1 / ((0.5γ)^2 + (ω+Ω)^2))
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
    osc_sx_init_state = chain(MPS(sites[1:n_osc_left-1], "0"),
                              parse_init_state_osc(
                                    sites[n_osc_left],
                                    parameters["left_oscillator_initial_state"]
                                   ))
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
                  plotsize=plotsize)

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
                  plotsize=plotsize)

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
                  plotsize=plotsize)

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
                  plotsize=plotsize)
                  
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
                  plotsize=plotsize)

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
                  plotsize=plotsize)

  savefig(plt, "spin_current.png")

  # Grafico dei coefficienti della chain map
  # ----------------------------------------
  plt = groupplot([1:p["number_of_oscillators_left"]
                   for p ∈ parameter_lists],
                  osc_chain_coefficients_left_super,
                  parameter_lists;
                  labels=["Ωᵢ" "κᵢ"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                             "oscillatori (sx)",
                  plotsize=plotsize)

  savefig(plt, "osc_left_coefficients.png")

  plt = groupplot([1:p["number_of_oscillators_right"]
                   for p ∈ parameter_lists],
                  osc_chain_coefficients_right_super,
                  parameter_lists;
                  labels=["Ωᵢ" "κᵢ"],
                  linestyles=[:solid :solid],
                  commonxlabel=L"i",
                  commonylabel="Coefficiente",
                  plottitle="Coefficienti della catena di "*
                            "oscillatori (dx)",
                  plotsize=plotsize)

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
                    plotsize=plotsize)

  savefig(plt, "snapshot.png")

  plt = unifiedplot(range_spins_super,
                    [occn[range_spins] for (occn, range_spins)
                     ∈ zip(snapshot_super, range_spins_super)],
                    parameter_lists;
                    linestyle=:solid,
                    xlabel=L"i",
                    ylabel="Numero di occupazione",
                    plottitle="Numeri di occupazione, alla fine, solo spin",
                    plotsize=plotsize)

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
                    plotsize=plotsize)

  savefig(plt, "normalisation.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
