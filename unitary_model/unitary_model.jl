#!/usr/bin/julia

using ITensors
using Plots
using Measures
using LaTeXStrings
using ProgressMeter
using LinearAlgebra
using JSON
using Base.Filesystem
using DataFrames
using CSV

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/tedopa.jl")

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
  #spin_current_super = []
  maxdim_monitor_super = []
  #spin_chain_levels_super = []
  #osc_levels_left_super = []
  #osc_levels_right_super = []
  #normalisation_super = []
  #hermiticity_monitor_super = []

  # Precaricamento
  # ==============
  # Se in tutte le liste di parametri il numero di siti è lo stesso, posso
  # definire qui una volta per tutte alcuni elementi "pesanti" che servono dopo.
  S_sites_list = [p["number_of_spin_sites"] for p in parameter_lists]
  L_sites_list = [p["number_of_oscillators_left"] for p in parameter_lists]
  R_sites_list = [p["number_of_oscillators_right"] for p in parameter_lists]
  if all(x -> x == first(S_sites_list), S_sites_list) &&
     all(x -> x == first(L_sites_list), L_sites_list) &&
     all(x -> x == first(R_sites_list), R_sites_list)
    preload = true
    n_spin_sites = first(S_sites_list)
    n_osc_left = first(L_sites_list)
    n_osc_right = first(R_sites_list)
    sites = [siteinds("Osc", n_osc_left);
             siteinds("S=1/2", n_spin_sites);
             siteinds("Osc", n_osc_right)]

    #=
    # - la corrente di spin
    # Prima costruisco gli operatori sulla catena di spin, poi li
    # estendo con l'identità sui restanti siti.
    spin_current_ops = [chain(MPS(sites[1:1], "vecid"),
                              j,
                              MPS(sites[end:end], "vecid"))
                        for j in spin_current_op_list(sites[2:end-1])]

    # - l'occupazione degli autospazi dell'operatore numero
    # Ad ogni istante proietto lo stato corrente sugli autostati
    # dell'operatore numero della catena di spin, vale a dire calcolo
    # tr(ρₛ Pₙ) dove ρₛ è la matrice densità ridotta della catena di spin
    # e Pₙ è il proiettore ortogonale sull'n-esimo autospazio di N
    num_eigenspace_projs = [chain(MPS(sites[1:1], "vecid"),
                                  level_subspace_proj(sites[2:end-1], n),
                                  MPS(sites[end:end], "vecid"))
                            for n=0:n_sites]

    # - l'occupazione dei livelli degli oscillatori
    osc_levels_projs_left = [chain(osc_levels_proj(sites[1], n),
                                   MPS(sites[2:end], "vecid"))
                             for n=1:osc_dim]
    osc_levels_projs_right = [chain(MPS(sites[1:end-1], "vecid"),
                                    osc_levels_proj(sites[end], n))
                              for n=1:osc_dim]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecid")
  =#
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
      sites = [siteinds("Osc", n_osc_left);
               siteinds("S=1/2", n_spin_sites);
               siteinds("Osc", n_osc_right)]
    end


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
    # Per semplicità assumo che n_osc_left, n_osc_right e n_spin_sites siano
    # TUTTI pari.
    links_odd = ITensor[]
    links_even = ITensor[]
    s1 = sites[1]
    s2 = sites[2]
    # Coppia di oscillatori all'estremo sinistro:
    h = Ωₗ[end]      * op("num", s1) * op("id", s2) +
        0.5Ωₗ[end-1] * op("id", s1) * op("num", s2) +
        κₗ[end] * op("a+", s1) * op("a-", s2) +
        κₗ[end] * op("a-", s1) * op("a+", s2)
    push!(links_odd, exp(-0.5im * τ * h))
    # Riparto dall'oscillatore #1 della catena a sinistra (è quello attaccato al
    # sito di spin) e la attraverso verso sinistra, a due a due:
    for i = 3:2:n_osc_left-1
      j = n_osc_left - i
      # (i,i+1) = (3,4), (5,6), …, (N-1,N)
      # (j,j+1) = (N-3,N-2), (N-5,N-4), …, (1,2)
      # (I coefficienti Ω e κ seguono una numerazione inversa!)
      s1 = sites[i]
      s2 = sites[i+1]
      h = 0.5Ωₗ[j+1] * op("num", s1) * op("id", s2) +
          0.5Ωₗ[j]   * op("id", s1) * op("num", s2) +
          κₗ[j] * op("a+", s1) * op("a-", s2) +
          κₗ[j] * op("a-", s1) * op("a+", s2)
      push!(links_odd, exp(-0.5im * τ * h))
    end
    for i = 2:2:n_osc_left-1
      j = n_osc_left - i # (vedi sopra)
      s1 = sites[i]
      s2 = sites[i+1]
      h = 0.5Ωₗ[j+1] * op("num", s1) * op("id", s2) +
          0.5Ωₗ[j]   * op("id", s1)  * op("num", s2) +
          κₗ[j] * op("a+", s1) * op("a-", s2) +
          κₗ[j] * op("a-", s1) * op("a+", s2)
      push!(links_even, exp(-im * τ * h))
    end
    # La coppia oscillatore-spin di sinistra ricade tra i link pari:
    so = sites[n_osc_left] # primo oscillatore a sx
    ss = sites[n_osc_left+1] # primo spin a sx
    h = 0.5*Ωₗ[1] * op("num", so) * op("id", ss) +
        0.5ε      * op("id", so)  * op("Sz", ss) +
        ηₗ * op("a+", so) * op("Sx", ss) + 
        ηₗ * op("a-", so) * op("Sx", ss)
    push!(links_even, exp(-im * τ * h))
    # Continuo con le coppie (dispari, pari) della catena di spin:
    for j = 1:2:n_spin_sites-1
      j += n_osc_left
      s1 = sites[j]
      s2 = sites[j+1]
      h = 0.5ε * op("Sz", s1) * op("id", s2) +
          0.5ε * op("id", s1) * op("Sz", s2) +
          -0.5 * op("S+", s1) * op("S-", s2) +
          -0.5 * op("S-", s1) * op("S+", s2)
      push!(links_odd, exp(-0.5im * τ * h))
    end
    # e poi con le coppie (pari, dispari):
    for j = 2:2:n_spin_sites-2
      j += n_osc_left
      s1 = sites[j]
      s2 = sites[j+1]
      h = 0.5ε * op("Sz", s1) * op("id", s2) +
          0.5ε * op("id", s1) * op("Sz", s2) +
          -0.5 * op("S+", s1) * op("S-", s2) +
          -0.5 * op("S-", s1) * op("S+", s2)
      push!(links_even, exp(-im * τ * h))
    end
    # La coppia oscillatore-spin di destra ricade anch'essa tra i link pari:
    ss = sites[n_osc_left+n_spin_sites] # ultimo spin (a dx)
    so = sites[n_osc_left+n_spin_sites+1] # primo oscillatore a dx
    h = 0.5ε *     op("Sz", ss) * op("id", so) +
        0.5Ωᵣ[1] * op("id", ss) * op("num", so) +
        ηᵣ * op("Sx", ss) * op("a+", so) + 
        ηᵣ * op("Sx", ss) * op("a-", so) 
    push!(links_even, exp(-im * τ * h))
    # Continuo con la serie di oscillatori a destra, prima le coppie dispari:
    for j = 1:2:n_osc_right
      s1 = sites[n_osc_left + n_spin_sites + j]
      s2 = sites[n_osc_left + n_spin_sites + j+1]
      h = 0.5Ωᵣ[j]   * op("num", s1) * op("id", s2) +
          0.5Ωᵣ[j+1] * op("id", s1)  * op("num", s2) +
          κᵣ[j] * op("a+", s1) * op("a-", s2) +
          κᵣ[j] * op("a-", s1) * op("a+", s2)
      push!(links_odd, exp(-0.5im * τ * h))
    end
    # e poi le coppie pari:
    for j = 2:2:n_osc_right-1
      s1 = sites[n_osc_left + n_spin_sites + j]
      s2 = sites[n_osc_left + n_spin_sites + j+1]
      h = 0.5Ωᵣ[j]   * op("num", s1) * op("id", s2) +
          0.5Ωᵣ[j+1] * op("id", s1)  * op("num", s2) +
          κᵣ[j] * op("a+", s1) * op("a-", s2) +
          κᵣ[j] * op("a-", s1) * op("a+", s2)
      push!(links_even, exp(-im * τ * h))
    end
    # Arrivo infine alla coppia di oscillatori più a destra (una coppia dispari):
    s1 = sites[end-1]
    s2 = sites[end]
    h = 0.5Ωᵣ[end-1] * op("num", s1) * op("id", s2) +
        Ωᵣ[end]      * op("id", s1)  * op("num", s2) +
        κᵣ[end] * op("a+", s1) * op("a-", s2) +
        κᵣ[end] * op("a-", s1) * op("a+", s2)
    push!(links_odd, exp(-0.5im * τ * h))

    evo = [links_odd; links_even; links_odd]

    # Osservabili da misurare
    # =======================
    if !preload
      #=
      # - la corrente di spin
      spin_current_ops = [chain(MPS(sites[1:1], "vecid"),
      j,
      MPS(sites[end:end], "vecid"))
      for j in spin_current_op_list(sites[2:end-1])]

      # - l'occupazione degli autospazi dell'operatore numero
      num_eigenspace_projs = [chain(MPS(sites[1:1], "vecid"),
      level_subspace_proj(sites[2:end-1], n),
      MPS(sites[end:end], "vecid"))
      for n=0:n_sites]

      # - l'occupazione dei livelli degli oscillatori
      osc_levels_projs_left = [chain(osc_levels_proj(sites[1], n),
      MPS(sites[2:end], "vecid"))
      for n=1:osc_dim]
      osc_levels_projs_right = [chain(MPS(sites[1:end-1], "vecid"),
      osc_levels_proj(sites[end], n))
      for n=1:osc_dim]

      # - la normalizzazione (cioè la traccia) della matrice densità
      full_trace = MPS(sites, "vecid")
      =#
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto
    osc_sx_init_state = MPS(sites[1:n_osc_left], "Emp")
    spin_init_state = MPS(sites[n_osc_left+1:n_osc_left+n_spin_sites],
                          ["Up"; repeat(["Dn"], n_spin_sites-1)])
    osc_dx_init_state = MPS(sites[end-n_osc_right+1:end], "Emp")
    current_state = chain(osc_sx_init_state,
                          spin_init_state,
                          osc_dx_init_state)

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [expect(current_state, "num")]
    maxdim_monitor = Int[maxlinkdim(current_state)]
    #spin_current = [[real(inner(j, current_state)) for j in spin_current_ops]]
    #chain_levels = [levels(num_eigenspace_projs, current_state)]
    #osc_levels_left = [levels(osc_levels_projs_left, current_state)]
    #osc_levels_right = [levels(osc_levels_projs_right, current_state)]
    #normalisation = [real(inner(full_trace, current_state))]
    #hermiticity_monitor = Real[0]
    #time_instants = Real[0]

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
        #=
        Calcolo dapprima la traccia della matrice densità. Se non devia
        eccessivamente da 1, in ogni caso influisce sul valore delle
        osservabili che calcolo successivamente, che si modificano dello
        stesso fattore, e devono essere quindi corrette di un fattore pari
        al reciproco della traccia.
        =#
        #trace = real(inner(full_trace, current_state))

        #push!(normalisation,
        #      trace)
        push!(occ_n,
              expect(current_state, "num"))
        #push!(spin_current,
        #      [real(inner(j, current_state)) for j in spin_current_ops] ./ trace)
        #push!(chain_levels,
        #      levels(num_eigenspace_projs, current_state) ./ trace)
        #push!(osc_levels_left,
        #      levels(osc_levels_projs_left, current_state) ./ trace)
        #push!(osc_levels_right,
        #      levels(osc_levels_projs_right, current_state) ./ trace)
        push!(maxdim_monitor,
              maxlinkdim(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([[Symbol("occ_n_left$n") for n∈n_osc_left:-1:1];
                                [Symbol("occ_n_spin$n") for n∈1:n_spin_sites];
                                [Symbol("occ_n_right$n") for n∈1:n_osc_right]])
      push!(dict, name => tmp_list[j,:])
    end
    #=
    tmp_list = hcat(spin_current...)
    for (j, name) in enumerate([Symbol("spin_current$n") for n = 1:n_sites-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(osc_levels_left...)
    for (j, name) in enumerate([Symbol("levels_left$n") for n = 0:osc_dim-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(chain_levels...)
    for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(osc_levels_right...)
    for (j, name) in enumerate([Symbol("levels_right$n") for n = 0:osc_dim:-1])
      push!(dict, name => tmp_list[j,:])
    end
    push!(dict, :full_trace => normalisation)
    push!(dict, :hermiticity => hermiticity_monitor)
    =#
    push!(dict, :maxdim => maxdim_monitor)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    #push!(spin_current_super, spin_current)
    #push!(chain_levels_super, chain_levels)
    #push!(osc_levels_left_super, osc_levels_left)
    #push!(osc_levels_right_super, osc_levels_right)
    push!(maxdim_monitor_super, maxdim_monitor)
    #push!(normalisation_super, normalisation)
    #push!(hermiticity_monitor_super, hermiticity_monitor)
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
                         displayed_sites=1:n_osc_left,
                         labels=string.(n_osc_left:-1:1),
                         linestyles=repeat([:solid], n_osc_left),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione "*
                                    "(oscillatori a sx)",
                         plot_size=plot_size)
  savefig(plt, "occ_n_osc_left.png")

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  start = n_osc_left
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=start+1:start+n_spin_sites,
                         labels=string.(1:n_spin_sites),
                         linestyles=repeat([:solid], n_spin_sites),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione (spin)",
                         plot_size=plot_size)
  savefig(plt, "occ_n_spins.png")
  
  # Grafico dei numeri di occupazione (oscillatori dx)
  # --------------------------------------------------
  start = n_osc_left + n_spin_sites
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=start+1:start+n_osc_right,
                         labels=string.(1:n_osc_right),
                         linestyles=repeat([:solid], n_osc_right),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione "*
                                    "(oscillatori dx)",
                         plot_size=plot_size)
  savefig(plt, "occ_n_osc_right.png")

  #=
  # Grafico dei numeri di occupazione (oscillatori + totale catena)
  # ---------------------------------------------------------------
  data_super = []
  for occ_n in occ_n_super
    data = []
    for row in occ_n
      push!(data,
            [first(row), sum(row[2:end-1]), last(row)])
    end
    push!(data_super, data)
  end
  #
  plt = plot_time_series(data_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["L", "catena", "R"],
                         linestyles=[:solid, :dot, :solid],
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione (oscillatori + totale catena)",
                         plot_size=plot_size
                        )
  savefig(plt, "occ_n_sums.png")
  =#

  # Grafico dei ranghi del MPS
  # --------------------------
  plt = plot_time_series(maxdim_monitor_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[nothing],
                         linestyles=[:solid],
                         x_label=L"\lambda\, t",
                         y_label=L"\max_k\,\chi_{k,k+1}",
                         plot_title="Rango massimo del MPS",
                         plot_size=plot_size
                        )
  savefig(plt, "maxdim_monitor.png")

  #=
  # Grafico della traccia della matrice densità
  # -------------------------------------------
  # Questo serve più che altro per controllare che rimanga sempre pari a 1.
  plt = plot_time_series(normalisation_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[nothing],
                         linestyles=[:solid],
                         x_label=L"\lambda\, t",
                         y_label=L"\operatorname{tr}\,\rho(t)",
                         plot_title="Normalizzazione della matrice densità",
                         plot_size=plot_size
                        )
  savefig(plt, "dm_normalisation.png")

  # Grafico della traccia della matrice densità
  # -------------------------------------------
  # Questo serve più che altro per controllare che rimanga sempre pari a 1.
  plt = plot_time_series(hermiticity_monitor_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[nothing],
                         linestyles=[:solid],
                         x_label=L"\lambda\, t",
                         y_label=L"\Vert\rho_\mathrm{L}(t)-\rho_\mathrm{L}(t)^\dagger\Vert",
                         plot_title="Controllo hermitianità della matrice densità",
                         plot_size=plot_size
                        )
  savefig(plt, "hermiticity_monitor.png")

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
  len = size(hcat(chain_levels_super[begin]...), 1) - 1
  plt = plot_time_series(chain_levels_super,
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

  # Grafico dell'occupazione dei livelli degli oscillatori
  # ------------------------------------------------------
  for (list, pos) in zip([osc_levels_left_super, osc_levels_right_super],
                         ["sx", "dx"])
    len = size(hcat(list[begin]...), 1) - 1
    plt = plot_time_series(list,
                           parameter_lists;
                           displayed_sites=nothing,
                           labels=[string.(0:len-1); "total"],
                           linestyles=[repeat([:solid], len); :dash],
                           x_label=L"\lambda\, t",
                           y_label=L"n",
                           plot_title="Occupazione degli autospazi "
                           * "dell'oscillatore $pos",
                           plot_size=plot_size
                          )
    savefig(plt, "osc_levels_$pos.png")
  end
  =#

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
