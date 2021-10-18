#!/usr/bin/julia

using ITensors
using Plots
using Measures
using LaTeXStrings
using ProgressMeter
using LinearAlgebra
using JSON

include("spin_chain_space.jl")
include("harmonic_oscillator_space.jl")

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let  
  # Imposta la cartella di base per l'output
  # ========================================
  src_path = Base.source_path()
  base_path = src_path[findlast(isequal('/'), src_path)+1:end]
  if base_path[end-2:end] == ".jl"
    base_path = base_path[begin:end-3]
  end
  base_path *= '/'

  # Lettura dei parametri della simulazione
  # =======================================
  input_files = ARGS
  parameter_lists = []
  for f in input_files
    open(f) do input
      s = read(input, String)
      push!(parameter_lists, JSON.parse(s))
    end
  end

  # Carico il dizionario dei "nomi brevi" per i parametri, che servirà
  # per i titoli nei grafici dopo
  f = open("short_names_dictionary.json", "r")
  short_name = JSON.parse(read(f, String))
  close(f)

  function construct_step_list(total_time, ε)
    # Siccome i risultati sono migliori se il numero di passi della simulazione
    # è proporzionale ad ε, ho bisogno di costruire l'array con tutti gli istanti
    # di tempo toccati dalla simulazione per ciascun file di parametri; tale
    # array serve anche in seguito per disegnare i grafici, quindi usando questa
    # funzione mi assicuro che l'array sia creato in modo consistente.
    n_steps = Int(total_time * ε)
    step_list = collect(LinRange(0, total_time, n_steps))
    return step_list
  end

  # Le seguenti liste conterranno i risultati della simulazione per ciascuna
  # lista di parametri fornita.
  occ_n_super = []
  current_super = []
  maxdim_monitor_super = []

  # Definisco qui alcune funzioni che verranno usate nel ciclo principale.
  # - funzioni per misurare la corrente di spin
  function current_1(i::Int, obs_index::Int)
    if i == obs_index
      str = "vecσy"
    elseif i == obs_index+1
      str = "vecσx"
    else
      str = "vecid"
    end
    return str
  end
  function current_2(i::Int, obs_index::Int)
    if i == obs_index
      str = "vecσx"
    elseif i == obs_index+1
      str = "vecσy"
    else
      str = "vecid"
    end
    return str
  end

  for parameters in parameter_lists
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
    time_step_list = construct_step_list(parameters["simulation_end_time"], ε)
    time_step = time_step_list[2] - time_step_list[1]

    avg_occ_n(T::Number, ω::Number) = 1 / (ℯ^(ω / T) - 1)

    # Costruzione della catena
    # ========================
    n_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
    sites = vcat(
      [Index(osc_dim^2, "vecOsc")],
      siteinds("vecS=1/2", n_sites),
      [Index(osc_dim^2, "vecOsc")]
    )

    # Stati di singola eccitazione
    single_ex_state_tags = [vcat(
      ["vecid"],
      [i == n ? "Up:Up" : "Dn:Dn" for i = 1:n_sites],
      ["vecid"]
    ) for n = 1:n_sites]
    single_ex_states = [MPS(sites, tags) for tags in single_ex_state_tags]

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
    #   Il metodo per calcolarla è questo, finché non mi viene in mente
    #   qualcosa di più comodo: l'operatore della corrente,
    #   J_k = λ/2 (σ ˣ⊗ σ ʸ-σ ʸ⊗ σ ˣ),
    #   viene separato nei suoi due addendi, che sono applicati allo
    #   stato corrente in tempi diversi; in seguito sottraggo il secondo
    #   al primo, e moltiplico tutto per λ/2 (che è 1/2).
    current_op_list_1 = [MPS(sites, vcat(
      ["Emp:Emp"],
      [current_1(i, k) for i = 1:n_sites],
      ["Emp:Emp"]
    )) for k = 1:n_sites-1]
    current_op_list_2 = [MPS(sites, vcat(
      ["Emp:Emp"],
      [current_2(i, k) for i = 1:n_sites],
      ["Emp:Emp"]
    )) for k = 1:n_sites-1]
    current_op_list = [0.5 * (J₊ - J₋) for (J₊,J₋) in zip(current_op_list_1, current_op_list_2)]
    # Ora per misurare la corrente basta prendere il prodotto interno con questi.

    # Simulazione
    # ===========
    # Stato iniziale: l'oscillatore sx è in equilibrio termico, il resto è vuoto
    # Se T == 0 invece parto con un'eccitazione nella catena (per creare una
    # situazione simile a quella della catena isolata).
    if T == 0
      current_state = single_ex_states[1]
    else
      current_state = MPS(vcat(
                               [state(sites[1], "ThermEq"; ω, T)],
                               [state(sites[1+j], "Dn:Dn") for j=1:n_sites],
                               [state(sites[end], "Emp:Emp")]
                              )
                         )
    end

    # Misuro le osservabili sullo stato iniziale
    occ_n = [[inner(s, current_state) for s in occ_n_list]]
    maxdim_monitor = Int[maxlinkdim(current_state)]
    current = [[real(inner(j, current_state)) for j in current_op_list]]

    # ...e si parte!
    progress = Progress(length(time_step_list), 1, "Simulazione in corso ", 20)
    for step in time_step_list[2:end]
      # Uso l'espansione di Trotter al 2° ordine
      current_state = apply(vcat(links_odd, links_even, links_odd), current_state, cutoff=max_err, maxdim=max_dim)
      #
      occ_n = vcat(occ_n, [[real(inner(s, current_state)) for s in occ_n_list]])
      current = vcat(current, [[real(inner(j, current_state)) for j in current_op_list]])
      push!(maxdim_monitor, maxlinkdim(current_state))
      next!(progress)
    end

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(current_super, current)
    push!(maxdim_monitor_super, maxdim_monitor)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  plotsize = Int(ceil(sqrt(length(parameter_lists)))) .* (600, 400)
  # Innanzitutto, analizzo l'array dei parametri e individuo quali parametri
  # variano tra un caso e l'altro. La funzione 'allunique' restituisce true se
  # tutti i valori nella lista sono distinti; creo dunque una lista per ciascun
  # tipo di parametro, e faccio il confronto.
  distinct_parameters = String[]
  for key in keys(parameter_lists[begin])
    test_list = [p[key] for p in parameter_lists]
    if !all(x -> x == first(test_list), test_list)
      push!(distinct_parameters, key)
    end
  end
  repeated_parameters = setdiff(keys(parameter_lists[begin]), distinct_parameters)

  function subplot_title(values_dict, keys)
    # Questa funzione costruisce il titolo personalizzato per ciascun sottografico,
    # che indica solo i parametri che cambiano da una simulazione all'altra
    return join(
      [short_name[k] * "=" * string(values_dict[k]) for k in keys],
      ", "
     )
  end
  function shared_title_fake_plot(subject::String)
    #= Siccome Plots.jl non ha ancora, che io sappia, un modo per dare un titolo
       a un gruppo di grafici, mi arrangio con questa soluzione trovata su
       StackOverflow, che consiste nel creare un grafico vuoto che contiene il
       titolo voluto come annotazione.
       Il titolone contiene i parametri comuni a tutte le simulazioni (perciò
       posso prendere senza problemi uno degli elementi di parameter_lists
       qualunque) e lo uso come titolo del gruppo di grafici.
    =#
    # Inserire in questo array i parametri che non si vuole che appaiano nel
    # titolo:
    hidden_parameters = ["simulation_end_time"]
    #
    shared_title = join(
      [short_name[k] * "=" * string(parameter_lists[begin][k]) for k in setdiff(repeated_parameters, hidden_parameters)],
      ", "
    )
    y = ones(3) # Dati falsi per far apparire un grafico
    title_fake_plot = Plots.scatter(y, marker=0, markeralpha=0, ticks=nothing, annotations=(2, y[2], text(subject * "\n" * shared_title)), axis=false, grid=false, leg=false, bottom_margin=2cm, size=(200,100))
    return title_fake_plot
  end

  function plot_time_series(data_super; labels, linestyles, x_label, y_label, plot_title)
    #= Disegna un grafico a partire dagli array "_super" ottenuti dalla
       simulazione qui sopra.

       Argomenti
       ---------
     · `data::Array{Any}`: un array di dimensione uguale a quella di
       parameter_lists.
       Ciascun elemento di data è un array X che rappresenta la evoluzione nel
       tempo (durante la simulazione) di N quantità: ogni riga per la
       precisione è una N-upla di valori che sono i valori di queste quantità
       ad ogni istante di tempo.
       L'array Xᵀ = hcat(X...) ne è la trasposizione: la sua riga j-esima, a
       cui accedo con Xᵀ[j,:], dà la serie temporale della j-esima quantità.

     · `labels::Array{String}`: un array di dimensione N che contiene le
       etichette da assegnare alla linea di ciascuna quantità da disegnare.

     · `linestyles::Array{Symbol}`: come `labels`, ma per gli stili delle linee.

     · `input_xlabel::String`: etichetta delle ascisse (comune a tutti)

     · `input_ylabel::String`: etichetta delle ordinate (comune a tutti)

     · `plot_title::String`: titolo grande del grafico
    =#
    subplots = []
    for (p, data) in zip(parameter_lists, data_super)
      time_step_list = construct_step_list(p["simulation_end_time"], p["spin_excitation_energy"])
      dataᵀ = hcat(data...)
      N = size(dataᵀ, 1)
      this_plot = plot()
      for j = 1:N
        plot!(this_plot, time_step_list,
                         dataᵀ[j,:],
                         label=labels[j],
                         linestyle=linestyles[j],
                         legend=:outerright,
                         left_margin=5mm,
                         bottom_margin=5mm
                        )
      end
      xlabel!(this_plot, x_label)
      ylabel!(this_plot, y_label)
      title!(this_plot, subplot_title(p, distinct_parameters))
      push!(subplots, this_plot)
    end
    group_plot = Plots.plot(
      shared_title_fake_plot(plot_title),
      Plots.plot(subplots..., layout=length(subplots), size=plotsize),
      layout=grid(2, 1, heights=[0.1, 0.9])
    )
    return group_plot
  end

  # Grafico dei numeri di occupazione
  # ---------------------------------
  len = size(hcat(occ_n_super[begin]...), 1)
  # È la lunghezza delle righe di vari array `occ_n`: per semplicità assumo che
  # siano tutti della stessa forma, altrimenti dovrei far calcolare alla
  # funzione `plot_time_series` anche tutto ciò che varia con tale lunghezza,
  # come `labels` e `linestyles`.
  #
  plt = plot_time_series(occ_n_super;
                         labels=vcat(["L"], string.(collect(1:len-2)), ["R"]),
                         linestyles=vcat([:dash], repeat([:solid], len-2), [:dash]),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione"
                        )
  savefig(plt, base_path * "occ_n.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  plt = plot_time_series(maxdim_monitor_super;
                         labels=["MPS"],
                         linestyles=[:solid],
                         x_label=L"\lambda\, t",
                         y_label=L"\max_k\,\chi_{k,k+1}",
                         plot_title="Rango massimo del MPS"
                        )
  savefig(plt, base_path * "maxdim_monitor.png")

  # Grafico della corrente di spin
  # ------------------------------
  len = size(hcat(current_super[begin]...), 1)
  plt = plot_time_series(current_super;
                         labels=string.(collect(1:len)),
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"j_{k,k+1}",
                         plot_title="Corrente di spin"
                        )
  savefig(plt, base_path * "spin_current.png")

  return
end
