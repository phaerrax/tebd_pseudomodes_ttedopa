#!/usr/bin/julia

using ITensors
using Plots
using Measures
using LaTeXStrings
using ProgressMeter
using LinearAlgebra
using JSON
using Base.Filesystem

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let  
  parameter_lists = load_parameters(ARGS)
  tot_sim_n = length(parameter_lists)

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  occ_n_super = []
  spin_current_super = []
  maxdim_monitor_super = []
  chain_levels_super = []
  osc_levels_left_super = []
  osc_levels_right_super = []

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
    γ = parameters["oscillator_damping_coefficient"]
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)

    # Costruzione della catena
    # ========================
    n_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
    sites = vcat(
      [Index(osc_dim^2, "vecOsc")],
      siteinds("vecS=1/2", n_sites),
      [Index(osc_dim^2, "vecOsc")]
    )

    # Stati di singola eccitazione
    single_ex_states = [chain(MPS(sites[1:1], "vecid"),
                              single_ex_state(sites[2:end-1], k),
                              MPS(sites[end:end], "vecid"))
                        for k = 1:n_sites]

    #= Definizione degli operatori nell'equazione di Lindblad
       ======================================================
       I siti del sistema sono numerati come segue:
       | 1 | 2 | ... | n_sites | n_sites+1 | n_sites+2 |
         ↑   │                        │          ↑
         │   └───────────┬────────────┘          │
         │               │                       │
         │        catena di spin                 │
       oscillatore sx                    oscillatore dx
       (Dato che non so come definire delle funzioni "op" che accettano Index di
       tipi diversi, vecS=1/2 e vecOsc in questo caso, non posso spostare anche
       questi due operatori ℓ_sx ed ℓ_dx nei file separati...)
    =#
    # - operatore per la coppia oscillatore-spin di sinistra
    sL = sites[1]
    s1 = sites[2]
    ℓ_sx = ω * op("H1loc", sL) * op("id:id", s1) +
           0.5ε * op("id:id", sL) * op("H1loc", s1) +
           im*κ * op("asum:id", sL) * op("σx:id", s1) +
           -im*κ* op("id:asum", sL)  * op("id:σx", s1) +
           γ * op("damping", sL; ω=ω, T=T) * op("id:id", s1)
    expℓ_sx = exp(0.5time_step * ℓ_sx)
    #
    # - e quello per la coppia oscillatore-spin di destra
    sn = sites[end-1]
    sR = sites[end]
    ℓ_dx = 0.5ε * op("H1loc", sn) * op("id:id", sR) +
           ω * op("id:id", sn) * op("H1loc", sR) +
           im*κ * op("σx:id", sn) * op("asum:id", sR) +
           -im*κ * op("id:σx", sn) * op("id:asum", sR) +
           γ * op("id:id", sn) * op("damping", sR; ω=ω, T=0)
    expℓ_dx = exp(0.5time_step * ℓ_dx)

    # Costruzione dell'operatore di evoluzione
    # ========================================
    links_odd = vcat(
      [expℓ_sx],
      [op("expHspin", sites[j], sites[j+1]; t=0.5time_step, ε=ε) for j = 3:2:n_sites],
      [expℓ_dx]
    )
    links_even = [op("expHspin", sites[j], sites[j+1]; t=time_step, ε=ε) for j = 2:2:n_sites+1]

    # Osservabili da misurare
    # =======================
    # - i numeri di occupazione: per gli spin della catena si prende il prodotto
    #   interno con gli elementi di single_ex_states già definiti; per gli
    #   oscillatori, invece, uso
    osc_num_sx = MPS(sites, vcat(
      ["vecnum"],
      repeat(["vecid"], n_sites),
      ["vecid"]
    ))
    osc_num_dx = MPS(sites, vcat(
      ["vecid"],
      repeat(["vecid"], n_sites),
      ["vecnum"]
    ))
    occ_n_list = vcat([osc_num_sx], single_ex_states, [osc_num_dx])

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

    # Simulazione
    # ===========
    # Stato iniziale: l'oscillatore sx è in equilibrio termico, il resto è vuoto
    # Se T == 0 invece parto con un'eccitazione nella catena (per creare una
    # situazione simile a quella della catena isolata).
    if T == 0
      current_state = MPS(sites,
                          vcat(
                               ["Emp:Emp"],
                               [i == 1 ? "Up:Up" : "Dn:Dn" for i = 1:n_sites],
                               ["Emp:Emp"]
                              )
                         )
    else
      current_state = MPS(vcat(
                               [state(sites[1], "ThermEq"; ω, T)],
                               [state(sites[1+j], "Dn:Dn") for j=1:n_sites],
                               [state(sites[end], "Emp:Emp")]
                              ))
    end

    # Misuro le osservabili sullo stato iniziale
    occ_n = [[inner(s, current_state) for s in occ_n_list]]
    maxdim_monitor = Int[maxlinkdim(current_state)]
    spin_current = [[real(inner(j, current_state)) for j in spin_current_ops]]
    chain_levels = [levels(num_eigenspace_projs, current_state)]
    osc_levels_left = [levels(osc_levels_projs_left, current_state)]
    osc_levels_right = [levels(osc_levels_projs_right, current_state)]

    # ...e si parte!
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    for _ in time_step_list[2:end]
      # Uso l'espansione di Trotter al 2° ordine
      current_state = apply(vcat(links_odd, links_even, links_odd), current_state, cutoff=max_err, maxdim=max_dim)
      #
      push!(occ_n,
            [real(inner(s, current_state)) for s in occ_n_list])
      push!(spin_current,
            [real(inner(j, current_state)) for j in spin_current_ops])
      push!(chain_levels,
            levels(num_eigenspace_projs, current_state))
      push!(osc_levels_left,
            levels(osc_levels_projs_left, current_state))
      push!(osc_levels_right,
            levels(osc_levels_projs_right, current_state))
      push!(maxdim_monitor,
            maxlinkdim(current_state))
      next!(progress)
    end

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(spin_current_super, spin_current)
    push!(chain_levels_super, chain_levels)
    push!(osc_levels_left_super, osc_levels_left)
    push!(osc_levels_right_super, osc_levels_right)
    push!(maxdim_monitor_super, maxdim_monitor)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  # Se il primo argomento da riga di comando è una cartella (che contiene
  # anche i file di parametri, così anche i grafici verranno salvati in tale
  # cartella.
  prev_dir = pwd()
  if isdir(ARGS[1])
    cd(ARGS[1])
  end

  plot_size = Int(ceil(sqrt(length(parameter_lists)))) .* (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  len = size(hcat(occ_n_super[begin]...), 1)
  # È la lunghezza delle righe di vari array `occ_n`: per semplicità assumo che
  # siano tutti della stessa forma, altrimenti dovrei far calcolare alla
  # funzione `plot_time_series` anche tutto ciò che varia con tale lunghezza,
  # come `labels` e `linestyles`.
  #
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=vcat(["L"], string.(1:len-2), ["R"]),
                         linestyles=vcat([:dash], repeat([:solid], len-2), [:dash]),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione",
                         plot_size=plot_size
                        )
  savefig(plt, "occ_n_all.png")

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=2:len+1,
                         labels=string.(1:len-2),
                         linestyles=repeat([:solid], len-2),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione (solo spin)",
                         plot_size=plot_size
                        )
  savefig(plt, "occ_n_spins_only.png")
  
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

  cd(prev_dir) # Ritorna alla cartella iniziale.
  return
end
