#!/usr/bin/julia

using ITensors
using Plots
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

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let
  parameter_lists = load_parameters(ARGS)

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  occ_n_super = []
  spin_current_super = []
  maxdim_monitor_super = []

  # Definizione degli operatori nell'equazione di Lindblad
  # ======================================================
  # Molti operatori sono già definiti in spin_chain_space.jl: rimangono quelli
  # particolari per questo sistema, che sono specifici di questo sistema e
  # perciò non ha senso definire nel file comune. Essi sono l'operatore per
  # la coppia (1,2)
  function ITensors.op(::OpName"expℓ_sx", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number, ξ::Number)
    L = ε * op("H1loc", s1) * op("id:id", s2) +
        0.5ε * op("id:id", s1) * op("H1loc", s2) +
        op("HspinInt", s1, s2) +
        ξ * op("damping", s1) * op("id:id", s2)
    return exp(t * L)
  end
  # e quello per la coppia (n-1,n)
  function ITensors.op(::OpName"expℓ_dx", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number, ξ::Number)
    L = 0.5ε * op("H1loc", s1) * op("id:id", s2) +
        ε * op("id:id", s1) * op("H1loc", s2) +
        op("HspinInt", s1, s2) +
        ξ * op("id:id", s1) * op("damping", s2)
    return exp(t * L)
  end

  for parameters in parameter_lists
    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["spin_damping_coefficient"]
    T = parameters["temperature"]

    # - intervallo temporale delle simulazioni
    time_step_list = construct_step_list(parameters["simulation_end_time"], ε)
    time_step = time_step_list[2] - time_step_list[1]

    # Costruzione della catena
    # ========================
    n_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
    # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
    sites = siteinds("vecS=1/2", n_sites)

    # Stati di singola eccitazione
    single_ex_states = MPS[productMPS(sites, n -> n == i ? "Up:Up" : "Dn:Dn") for i = 1:n_sites]

    # Costruzione dell'operatore di evoluzione
    # ========================================
    ξL = T==0 ? κ : κ * (1 + 2 / (ℯ^(ε/T) - 1))
    ξR = κ

    links_odd = vcat(
      [op("expℓ_sx", sites[1], sites[2]; t=0.5time_step, ε=ε, ξ=ξL)],
      [op("expHspin", sites[j], sites[j+1]; t=0.5time_step, ε=ε) for j = 3:2:n_sites-3],
      [op("expℓ_dx", sites[n_sites-1], sites[n_sites]; t=0.5time_step, ε=ε, ξ=ξR)]
    )
    links_even = [op("expHspin", sites[j], sites[j+1]; t=time_step, ε=ε) for j = 2:2:n_sites-2]

    # Osservabili da misurare
    # =======================
    # - la corrente di spin
    spin_current_ops = spin_current_op_list(sites)

    # Simulazione
    # ===========
    # Stato iniziale: c'è un'eccitazione nel primo sito
    current_state = single_ex_states[1]

    # Misuro le osservabili sullo stato iniziale
    occ_n = [[inner(s, current_state) for s in single_ex_states]]
    maxdim_monitor = Int[maxlinkdim(current_state)]
    spin_current = [[real(inner(j, current_state)) for j in spin_current_ops]]

    # ...e si parte!
    progress = Progress(length(time_step_list), 1, "Simulazione in corso ", 20)
    for step in time_step_list[2:end]
      current_state = apply(vcat(links_odd, links_even, links_odd), current_state, cutoff=max_err, maxdim=max_dim)
      #
      occ_n = vcat(occ_n, [[real(inner(s, current_state)) for s in single_ex_states]])
      spin_current = vcat(spin_current, [[real(inner(j, current_state)) for j in spin_current_ops]])
      push!(maxdim_monitor, maxlinkdim(current_state))
      next!(progress)
    end

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(spin_current_super, spin_current)
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
                         labels=string.(1:len),
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione",
                         plot_size=plot_size
                        )
  savefig(plt, "occ_n_all.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  plt = plot_time_series(maxdim_monitor_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=["MPS"],
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

  cd(prev_dir) # Ritorna alla cartella iniziale.
  return
end
