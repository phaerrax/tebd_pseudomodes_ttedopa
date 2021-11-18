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
  normalisation_super = []
  hermiticity_monitor_super = []

  # Precaricamento
  # ==============
  # Se in tutte le liste di parametri il numero di siti è lo stesso, posso
  # definire qui una volta per tutte alcuni elementi "pesanti" che servono dopo.
  n_sites_list = [p["number_of_spin_sites"] for p in parameter_lists]
  if all(x -> x == first(n_sites_list), n_sites_list)
    preload = true
    n_sites = first(n_sites_list)
    sites = vcat(
                 [Index(osc_dim^2, "vecOsc")],
                 siteinds("vecS=1/2", n_sites),
                 [Index(osc_dim^2, "vecOsc")]
                )
    single_ex_states = [chain(MPS(sites[1:1], "vecid"),
                              single_ex_state(sites[2:end-1], k),
                              MPS(sites[end:end], "vecid"))
                        for k = 1:n_sites]
    # - i numeri di occupazione: per gli spin della catena si prende il prodotto
    #   interno con gli elementi di single_ex_states già definiti; per gli
    #   oscillatori, invece, uso
    osc_num_sx = MPS(sites, vcat(["vecnum"],
                                 repeat(["vecid"], n_sites),
                                 ["vecid"]))
    osc_num_dx = MPS(sites, vcat(["vecid"],
                                 repeat(["vecid"], n_sites),
                                 ["vecnum"]))
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

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecid")
  else
    preload = false
  end

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

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)

    # Costruzione della catena
    # ========================
    if !preload
      n_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
      sites = vcat(
                   [Index(osc_dim^2, "vecOsc")],
                   siteinds("vecS=1/2", n_sites),
                   [Index(osc_dim^2, "vecOsc")]
                  )
      single_ex_states = [chain(MPS(sites[1:1], "vecid"),
                                single_ex_state(sites[2:end-1], k),
                                MPS(sites[end:end], "vecid"))
                          for k = 1:n_sites]
    end

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
           γₗ * op("damping", sL; ω=ω, T=T) * op("id:id", s1)
    # - e quello per la coppia oscillatore-spin di destra
    sn = sites[end-1]
    sR = sites[end]
    ℓ_dx = 0.5ε * op("H1loc", sn) * op("id:id", sR) +
           ω * op("id:id", sn) * op("H1loc", sR) +
           im*κ * op("σx:id", sn) * op("asum:id", sR) +
           -im*κ * op("id:σx", sn) * op("id:asum", sR) +
           γᵣ * op("id:id", sn) * op("damping", sR; ω=ω, T=0)
    #
    function links_odd(τ)
      return [exp(τ * ℓ_sx);
              [op("expHspin", sites[j], sites[j+1]; t=τ, ε=ε) for j = 3:2:n_sites];
              exp(τ * ℓ_dx)]
    end
    function links_even(τ)
      return [op("expHspin", sites[j], sites[j+1]; t=τ, ε=ε) for j = 2:2:n_sites+1]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             time_step,
                             parameters["TS_expansion_order"])

    # Osservabili da misurare
    # =======================
    if !preload
      # - i numeri di occupazione
      osc_num_sx = MPS(sites, vcat(["vecnum"],
                                   repeat(["vecid"], n_sites),
                                   ["vecid"]))
      osc_num_dx = MPS(sites, vcat(["vecid"],
                                   repeat(["vecid"], n_sites),
                                   ["vecnum"]))
      occ_n_list = vcat([osc_num_sx], single_ex_states, [osc_num_dx])

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
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    osc_sx_init_state = MPS([state(sites[1], "ThermEq"; ω, T)])
    osc_dx_init_state = MPS([state(sites[end], "Emp:Emp")])
    current_state = chain(osc_sx_init_state,
                          parse_init_state(sites[2:end-1],
                                           parameters["chain_initial_state"]),
                          osc_dx_init_state)

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [[inner(s, current_state) for s in occ_n_list]]
    maxdim_monitor = Int[maxlinkdim(current_state)]
    spin_current = [[real(inner(j, current_state)) for j in spin_current_ops]]
    chain_levels = [levels(num_eigenspace_projs, current_state)]
    osc_levels_left = [levels(osc_levels_projs_left, current_state)]
    osc_levels_right = [levels(osc_levels_projs_right, current_state)]
    normalisation = [real(inner(full_trace, current_state))]
    hermiticity_monitor = Real[0]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state,
                            cutoff=max_err,
                            maxdim=max_dim)
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
      push!(spin_current,
            [real(inner(j, current_state)) for j in spin_current_ops] ./ trace)
      push!(chain_levels,
            levels(num_eigenspace_projs, current_state) ./ trace)
      push!(osc_levels_left,
            levels(osc_levels_projs_left, current_state) ./ trace)
      push!(osc_levels_right,
            levels(osc_levels_projs_right, current_state) ./ trace)
      push!(maxdim_monitor,
            maxlinkdim(current_state))

      #=
      Controllo che la matrice densità ridotta dell'oscillatore a sinistra
      sia una valida matrice densità: hermitiana e semidefinita negativa.
      Calcolo la traccia parziale su tutti i siti tranne il primo, ricreo
      la matrice a partire dal vettore, e faccio i dovuti controlli.
      Non so come creare un MPO misto di matrici e vettori, quindi creo osc_dim²
      operatori che estraggono tutte le coordinate del vettore.
      =#
      mat = Array{Complex}(undef, osc_dim, osc_dim)
      for j = 1:osc_dim, k = 1:osc_dim
        proj = chain(MPS([state(sites[1], "mat_comp"; j, k)]),
                     MPS(sites[2:end], "vecid"))
        mat[j, k] = inner(proj, current_state)
      end
      # Avverti solo se la matrice non è semidefinita positiva. Per calcolare
      # la positività degli autovalori devo tagliare via la loro parte reale,
      # praticamente assumendo che siano reali (cioè che mat sia hermitiana).
      if any(x -> x < 0, real.(eigvals(mat)))
        @warn "La matrice densità del primo sito non è semidefinita positiva."
      end
      diff = sqrt(norm(mat - mat'))
      push!(hermiticity_monitor,
            diff)

      next!(progress)
    end
    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => construct_step_list(parameters))
    tmp_list = hcat(occ_n...)
    for (j, name) in enumerate([:occ_n_left;
                              [Symbol("occ_n_spin$n") for n = 1:n_sites];
                              :occ_n_right])
      push!(dict, name => tmp_list[j,:])
    end
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
    push!(dict, :maxdim => maxdim_monitor)
    push!(dict, :full_trace => normalisation)
    push!(dict, :hermiticity => hermiticity_monitor)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(spin_current_super, spin_current)
    push!(chain_levels_super, chain_levels)
    push!(osc_levels_left_super, osc_levels_left)
    push!(osc_levels_right_super, osc_levels_right)
    push!(maxdim_monitor_super, maxdim_monitor)
    push!(normalisation_super, normalisation)
    push!(hermiticity_monitor_super, hermiticity_monitor)
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

  plot_size = (2, 0.5 + ceil(length(parameter_lists)/2)) .* (600, 400)

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

  cd(prev_dir) # Ritorna alla cartella iniziale.
  return
end
