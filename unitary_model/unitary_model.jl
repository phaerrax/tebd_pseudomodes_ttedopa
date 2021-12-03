#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/tedopa.jl")
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
  occ_n_super = []
  spin_current_super = []
  bond_dimensions_super = []
  spin_chain_levels_super = []
  osc_chain_coefficients_left_super = []
  osc_chain_coefficients_right_super = []
  snapshot_super = []

  # Precaricamento
  # ==============
  # Se in tutte le liste di parametri il numero di siti è lo stesso, posso
  # definire qui una volta per tutte alcuni elementi "pesanti" che servono dopo.
  S_sites_list = [p["number_of_spin_sites"] for p in parameter_lists]
  osc_dim_list = [p["oscillator_space_dimension"] for p in parameter_lists]
  L_sites_list = [p["number_of_oscillators_left"] for p in parameter_lists]
  R_sites_list = [p["number_of_oscillators_right"] for p in parameter_lists]
  if allequal(S_sites_list) &&
     allequal(osc_dim_list) &&
     allequal(L_sites_list) &&
     allequal(R_sites_list)
    preload = true
    n_spin_sites = first(S_sites_list)
    osc_dim = first(osc_dim_list)
    n_osc_left = first(L_sites_list)
    n_osc_right = first(R_sites_list)
    sites = [siteinds("Osc", n_osc_left; dim=osc_dim);
             siteinds("S=1/2", n_spin_sites);
             siteinds("Osc", n_osc_right; dim=osc_dim)]

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    # - la corrente di spin
    list = spin_current_op_list(sites[range_spins])
    spin_current_ops = [embed_slice(sites, range_spins, j)
                        for j in list]
    # - l'occupazione degli autospazi dell'operatore numero
    # Ad ogni istante proietto lo stato corrente sugli autostati
    # dell'operatore numero della catena di spin, vale a dire calcolo
    # (ψ,Pₙψ) dove ψ è lo stato corrente e Pₙ è il proiettore ortogonale
    # sull'n-esimo autospazio di N.
    spin_range = n_osc_left+1:n_osc_left+n_spin_sites
    projectors = [level_subspace_proj(sites[spin_range], n)
                  for n = 0:n_spin_sites]
    num_eigenspace_projs = [embed_slice(sites, spin_range, p)
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
    T = parameters["temperature"]
    γ = parameters["spectral_density_half_width"]
    ωc = parameters["frequency_cutoff"]
    osc_dim = parameters["oscillator_space_dimension"]

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
      sites = [siteinds("Osc", n_osc_left; dim=osc_dim);
               siteinds("S=1/2", n_spin_sites);
               siteinds("Osc", n_osc_right; dim=osc_dim)]
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
    J(ω) = γ/π * (1 / (γ^2 + (ω-Ω)^2) - 1 / (γ^2 + (ω+Ω)^2))
    Jtherm = ω -> thermalisedJ(J, ω, T, (-ωc, ωc))
    Jzero  = ω -> thermalisedJ(J, ω, 0, (0, ωc))
    (Ωₗ, κₗ, ηₗ) = chainmap_coefficients(Jtherm,
                                         (-ωc, ωc),
                                         n_osc_left;
                                         Nquad=nquad)
    (Ωᵣ, κᵣ, ηᵣ) = chainmap_coefficients(Jzero,
                                         (0, ωc),
                                         n_osc_right;
                                         Nquad=nquad)

    # Raccolgo i coefficienti in due array (uno per quelli a sx, l'altro per
    # quelli a dx) per poterli disegnare assieme nei grafici.
    # (I coefficienti κ sono uno in meno degli Ω! Per ora pareggio le lunghezze
    # inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ωₗ [0; κₗ]]
    osc_chain_coefficients_right = [Ωᵣ [0; κᵣ]]

    # Il seguente vettore contiene i coefficienti dei termini di singolo sito
    # dell'Hamiltoniano: moltiplico gli estremi per 2 in modo da compensare il
    # fattore 0.5 che si prende ciascun termine locale di H, dato che deve
    # essere condiviso tra due operatori (uno della serie pari, uno della
    # serie dispari).
    localcf = [reverse(Ωₗ); repeat([ε], n_spin_sites); Ωᵣ]
    localcf[begin] *= 2
    localcf[end] *= 2
    # Idem ma per i termini di interazione, a due siti; il j° elemento è
    # il coefficiente del termine (j,j+1).
    interactioncf = [reverse(κₗ); ηₗ; repeat([1], n_spin_sites-1); ηᵣ; κᵣ]

    hlist = ITensor[]
    for j ∈ 1:length(sites)-1
      s1 = sites[j]
      s2 = sites[j+1]
      h = 0.5localcf[j] * Hlocal(s1) * op("Id", s2) +
          0.5localcf[j+1] * op("Id", s1) * Hlocal(s2) +
          interactioncf[j] * Hinteraction(s1, s2)
      push!(hlist, h)
    end

    function links_odd(τ)
      return [(exp(-im * τ * h)) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [(exp(-im * τ * h)) for h in hlist[2:2:end]]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             τ,
                             parameters["TS_expansion_order"])

    # Osservabili da misurare
    # =======================
    if !preload
      # - la corrente di spin
      list = spin_current_op_list(sites[range_spins])
      spin_current_ops = [embed_slice(sites, range_spins, j)
                          for j in list]
      # - l'occupazione degli autospazi dell'operatore numero
      projectors = [level_subspace_proj(sites[range_spins], n)
                    for n = 0:n_spin_sites]
      num_eigenspace_projs = [embed_slice(sites, range_spins, p)
                              for p in projectors]
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto
    osc_sx_init_state = MPS(sites[range_osc_left], "0")
    spin_init_state = parse_init_state(sites[spin_range],
                                       parameters["chain_initial_state"])
    osc_dx_init_state = MPS(sites[range_osc_right], "0")
    current_state = chain(osc_sx_init_state,
                          spin_init_state,
                          osc_dx_init_state)

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [expect(current_state, "N")]
    bond_dimensions = [linkdims(current_state)]
    spin_current = [[real(inner(current_state, j * current_state))
                     for j in spin_current_ops]]
    spin_chain_levels = [levels(num_eigenspace_projs,
                                current_state)]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state;
                            cutoff=max_err,
                            maxdim=max_dim)
      if skip_count % skip_steps == 0
        push!(occ_n,
              expect(current_state, "N"))
        push!(spin_current,
              [real(inner(current_state, j * current_state))
               for j in spin_current_ops])
        push!(spin_chain_levels,
              levels(num_eigenspace_projs, current_state))
        push!(bond_dimensions,
              linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    snapshot = occ_n[end]

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([[Symbol("occ_n_left$n") for n∈n_osc_left:-1:1];
                                [Symbol("occ_n_spin$n") for n∈1:n_spin_sites];
                                [Symbol("occ_n_right$n") for n∈1:n_osc_right]])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(spin_current...)
    for (j, name) in enumerate([Symbol("spin_current$n")
                                for n = 1:n_spin_sites-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(spin_chain_levels...)
    for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(bond_dimensions...)
    len = n_osc_left + n_spin_sites + n_osc_right
    for (j, name) in enumerate([Symbol("bond_dim$n") for n ∈ 1:len-1])
      push!(dict, name => tmp_list[j,:])
    end
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(spin_current_super, spin_current)
    push!(spin_chain_levels_super, spin_chain_levels)
    push!(bond_dimensions_super, bond_dimensions)
    push!(osc_chain_coefficients_left_super, osc_chain_coefficients_left)
    push!(osc_chain_coefficients_right_super, osc_chain_coefficients_right)
    push!(snapshot_super, snapshot)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  plot_size = (2, 0.5 + ceil(length(parameter_lists)/2)) .* (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  # Grafico dei numeri di occupazione (oscillatori sx)
  # --------------------------------------------------
  len = size(hcat(occ_n_super[begin]...), 1)
  # È la lunghezza delle righe di vari array `occ_n`: per semplicità assumo che
  # siano tutti della stessa forma, altrimenti dovrei far calcolare alla
  # funzione `plot_time_series` anche tutto ciò che varia con tale lunghezza,
  # come `labels` e `linestyles`.
  #
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=range_osc_left,
                         labels=string.(reverse(range_osc_left)),
                         linestyles=repeat([:solid], n_osc_left),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione "*
                                    "(oscillatori a sx)",
                         plot_size=plot_size)
  savefig(plt, "occ_n_osc_left.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=range_spins,
                         labels=string.(1:n_spin_sites),
                         linestyles=repeat([:solid], n_spin_sites),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione (spin)",
                         plot_size=plot_size)
  savefig(plt, "occ_n_spins.png")
  
  # Grafico dei numeri di occupazione (oscillatori dx)
  # --------------------------------------------------
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=range_osc_right,
                         labels=string.(1:n_osc_right),
                         linestyles=repeat([:solid], n_osc_right),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione "*
                                    "(oscillatori dx)",
                         plot_size=plot_size)
  savefig(plt, "occ_n_osc_right.png")

  # Grafico dei numeri di occupazione (tot oscillatori + tot catena)
  # ----------------------------------------------------------------
  data_super = []
  for occ_n in occ_n_super
    data = []
    for row in occ_n
      push!(data, [sum(row[range_osc_left]),
                   sum(row[range_spins]),
                   sum(row[range_osc_right]),
                   sum(row)])
    end
    push!(data_super, data)
  end
  #
  plt = plot_time_series(data_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["osc. sx", "catena", "osc. dx", "tutti"],
                         linestyles=repeat([:solid], 4),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione (sommati)",
                         plot_size=plot_size
                        )
  savefig(plt, "occ_n_sums.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  len = size(hcat(bond_dimensions_super[begin]...), 1)
  plt = plot_time_series(bond_dimensions_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["($j,$(j+1))" for j=1:len],
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"\chi_{k,k+1}",
                         plot_title="Ranghi del MPS",
                         plot_size=plot_size
                        )
  savefig(plt, "bond_dimensions.png")

  # Grafico della corrente di spin
  # ------------------------------
  len = size(hcat(spin_current_super[begin]...), 1)
  plt = plot_time_series(spin_current_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["($j,$(j+1))" for j=1:len],
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"j_{k,k+1}",
                         plot_title="Corrente di spin",
                         plot_size=plot_size
                        )
  savefig(plt, "spin_current.png")
 
  # Grafico dell'occupazione degli autospazi di N della catena di spin
  # ------------------------------------------------------------------
  # L'ultimo valore di ciascuna riga rappresenta la somma di tutti i
  # restanti valori.
  len = size(hcat(spin_chain_levels_super[begin]...), 1) - 1
  plt = plot_time_series(spin_chain_levels_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[string.(0:len-1); "total"],
                         linestyles=[repeat([:solid], len); :dash],
                         x_label=L"\lambda\, t",
                         y_label=L"n",
                         plot_title="Occupazione degli autospazi "
                         * "della catena di spin",
                         plot_size=plot_size
                        )
  savefig(plt, "chain_levels.png")
 
  # Grafico dei coefficienti della chain map
  # ----------------------------------------
  plt = plot_standalone(osc_chain_coefficients_left_super,
                        parameter_lists;
                        labels=[L"\Omega_i", L"\kappa_i"],
                        linestyles=[:solid, :solid],
                        x_label=L"i",
                        y_label="Coefficiente",
                        plot_title="Coefficienti della catena di "*
                                   "oscillatori (sx)",
                        plot_size=plot_size
                        )
  savefig(plt, "osc_left_coefficients.png")

  plt = plot_standalone(osc_chain_coefficients_right_super,
                        parameter_lists;
                        labels=[L"\Omega_i", L"\kappa_i"],
                        linestyles=[:solid, :solid],
                        x_label=L"i",
                        y_label="Coefficiente",
                        plot_title="Coefficienti della catena di "*
                                   "oscillatori (dx)",
                        plot_size=plot_size
                        )
  savefig(plt, "osc_right_coefficients.png")

  # Istantanea dei numeri di occupazione alla fine
  # ----------------------------------------------
  plt = plot_standalone(snapshot_super,
                        parameter_lists;
                        labels=[nothing],
                        linestyles=[:solid],
                        x_label=L"i",
                        y_label="Numero di occupazione",
                        plot_title="Numeri di occupazione alla fine",
                        plot_size=plot_size
                        )
  savefig(plt, "snapshot.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
