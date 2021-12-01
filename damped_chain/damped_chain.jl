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
  chain_levels_super = []

  # Definizione degli operatori nell'equazione di Lindblad
  # ======================================================
  # Molti operatori sono già definiti in spin_chain_space.jl: rimangono quelli
  # particolari per questo sistema, che sono specifici di questo sistema e
  # perciò non ha senso definire nel file comune. Essi sono l'operatore per
  # la coppia (1,2)
  function ITensors.op(::OpName"expℓ_sx", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number, ξ::Number)
    L = ε * op("H1loc", s1) * op("Id:Id", s2) +
        0.5ε * op("Id:Id", s1) * op("H1loc", s2) +
        op("HspinInt", s1, s2) +
        ξ * op("damping", s1) * op("Id:Id", s2)
    return exp(t * L)
  end
  # e quello per la coppia (n-1,n)
  function ITensors.op(::OpName"expℓ_dx", ::SiteType"vecS=1/2", s1::Index, s2::Index; t::Number, ε::Number, ξ::Number)
    L = 0.5ε * op("H1loc", s1) * op("Id:Id", s2) +
        ε * op("Id:Id", s1) * op("H1loc", s2) +
        op("HspinInt", s1, s2) +
        ξ * op("Id:Id", s1) * op("damping", s2)
    return exp(t * L)
  end

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["spin_damping_coefficient"]
    T = parameters["temperature"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
    # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
    sites = siteinds("vecS=1/2", n_sites)

    # Stati di singola eccitazione
    single_ex_states = [single_ex_state(sites, j) for j = 1:n_sites]

    # Costruzione dell'operatore di evoluzione
    # ========================================
    ξL = T==0 ? κ : κ * (1 + 2 / (ℯ^(ε/T) - 1))
    ξR = κ

    function links_odd(τ)
      return [op("expℓ_sx", sites[1], sites[2]; t=τ, ε=ε, ξ=ξL);
              [op("expHspin", sites[j], sites[j+1]; t=τ, ε=ε) for j = 3:2:n_sites-3];
              op("expℓ_dx", sites[n_sites-1], sites[n_sites]; t=τ, ε=ε, ξ=ξR)]
    end
    function links_even(τ)
      return [op("expHspin", sites[j], sites[j+1]; t=τ, ε=ε)
              for j = 2:2:n_sites-2]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             time_step,
                             parameters["TS_expansion_order"])

    # Osservabili da misurare
    # =======================
    # - la corrente di spin
    spin_current_ops = spin_current_op_list(sites)

    # - l'occupazione degli autospazi dell'operatore numero
    # Ad ogni istante proietto lo stato corrente sugli autostati
    # dell'operatore numero della catena di spin, vale a dire calcolo
    # tr(ρₛ Pₙ) dove ρₛ è la matrice densità ridotta della catena di spin
    # e Pₙ è il proiettore ortogonale sull'n-esimo autospazio di N
    num_eigenspace_projs = [level_subspace_proj(sites, n) for n=0:n_sites]

    # Simulazione
    # ===========
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    current_state = parse_init_state(sites, parameters["chain_initial_state"])

    # Misuro le osservabili sullo stato iniziale
    occ_n = [[inner(s, current_state) for s in single_ex_states]]
    bond_dimensions = [linkdims(current_state)]
    spin_current = [[real(inner(j, current_state)) for j in spin_current_ops]]
    chain_levels = [levels(num_eigenspace_projs, current_state)]

    # ...e si parte!
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state,
                            cutoff=max_err,
                            maxdim=max_dim)
      #
      if skip_count % skip_steps == 0
        push!(occ_n,
              [real(inner(s, current_state)) for s in single_ex_states])
        push!(spin_current,
              [real(inner(j, current_state)) for j in spin_current_ops])
        push!(chain_levels,
              levels(num_eigenspace_projs, current_state))
        push!(bond_dimensions,
              linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list[1:skip_steps:end])
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([Symbol("occ_n_spin$n") for n = 1:n_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(spin_current...)
    for (j, name) in enumerate([Symbol("spin_current$n") for n = 1:n_sites-1])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(chain_levels...)
    for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_sites])
      push!(dict, name => tmp_list[j,:])
    end
    tmp_list = hcat(bond_dimensions...)
    for (j, name) in enumerate([Symbol("bond_dim$n") for n = 1:n_sites-1])
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
    push!(chain_levels_super, chain_levels)
    push!(bond_dimensions_super, bond_dimensions)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
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
  len = size(hcat(chain_levels_super[begin]...), 1) - 1
  plt = plot_time_series(chain_levels_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=[string.(0:len-1); "total"],
                         linestyles=[repeat([:solid], len); :dash],
                         x_label=L"\lambda\, t",
                         y_label=L"n",
                         plot_title="Occupazione degli autospazi "
                         * "dell'operatore numero",
                         plot_size=plot_size
                        )
  savefig(plt, "chain_levels.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
