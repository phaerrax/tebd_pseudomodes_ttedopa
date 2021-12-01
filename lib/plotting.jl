using Base.Filesystem
using Plots
using JSON

# Grafici
# =======
function categorise_parameters(parameter_lists)
  # Analizzo l'array dei parametri e individuo quali parametri variano tra un
  # caso e l'altro, suddividendoli tra "distinti" se almeno uno dei parametri
  # è diverso tra le simulazioni, e "ripetuti" se sono invece tutti uguali.
  distinct = String[]
  for key in keys(parameter_lists[begin])
    test_list = [p[key] for p in parameter_lists]
    if !allequal(test_list)
      push!(distinct, key)
    end
  end
  repeated = setdiff(keys(parameter_lists[begin]), distinct)
  return distinct, repeated
end
#
function subplot_title(values_dict, keys)
  # Questa funzione costruisce il titolo personalizzato per ciascun sottografico,
  # che indica solo i parametri che cambiano da una simulazione all'altra
  #
  f = open(lib_path * "/short_names_dictionary.json", "r")
  # Carico il dizionario dei "nomi brevi" per i parametri.
  short_name = JSON.parse(read(f, String))
  close(f)
  hidden_parameters = ["simulation_end_time", "skip_steps", "filename"]
  return join([short_name[k] * "=" * string(values_dict[k])
               for k in setdiff(keys, hidden_parameters)],
              ", ")
end
#
function shared_title_fake_plot(subject::String, parameters)
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
  hidden_parameters = ["simulation_end_time", "skip_steps", "filename"]
  _, repeated_parameters = categorise_parameters(parameters)
  #
  f = open(lib_path * "/short_names_dictionary.json", "r")
  # Carico il dizionario dei "nomi brevi" per i parametri.
  short_name = JSON.parse(read(f, String))
  close(f)
  #
  shared_title = join([short_name[k] * "=" * string(parameters[begin][k])
                       for k in setdiff(repeated_parameters, hidden_parameters)],
                      ", ")
  y = ones(3) # Dati falsi per far apparire un grafico
  title_fake_plot = Plots.scatter(y, marker=0, markeralpha=0, ticks=nothing, annotations=(2, y[2], text(subject * "\n" * shared_title)), axis=false, grid=false, leg=false, bottom_margin=2cm, size=(200,100))
  return title_fake_plot
end
#
function plot_time_series(data_super, parameter_super; displayed_sites, labels, linestyles, x_label, y_label, plot_title, plot_size)
  #= Disegna un grafico dei dati di un array "_super" al variare del tempo,
     ottenuti dalle simulazioni associate ai vari insiemi di parametri
     in `parameter_super`, raggruppando in un unica immagine i grafici
     relativi alla stessa grandezza.

     Argomenti
     ---------
   · `data_super::Array{Any}`: un array di dimensione uguale a quella di
     parameter_lists.
     Ciascun elemento di data è un array X che rappresenta la evoluzione nel
     tempo (durante la simulazione) di N quantità: ogni riga per la
     precisione è una N-upla di valori che sono i valori di queste quantità
     ad ogni istante di tempo.
     L'array Xᵀ = hcat(X...) ne è la trasposizione: la sua riga j-esima, a
     cui accedo con Xᵀ[j,:], dà la serie temporale della j-esima quantità.

   · `displayed_sites`: un array di M ≤ N indici in 1:N che indica quali
     serie temporali (cioè, riferite a quali siti) disegnare; il valore
     `nothing` implica che tutte vanno disegnate. Va bene anche un oggetto
     del tipo a:b, verrà convertito automaticamente

   · `labels::Array{String}`: un array di dimensione M che contiene le
     etichette da assegnare alla linea di ciascuna quantità da disegnare.

   · `linestyles::Array{Symbol}`: come `labels`, ma per gli stili delle linee.

   · `input_xlabel::String`: etichetta delle ascisse (comune a tutti)

   · `input_ylabel::String`: etichetta delle ordinate (comune a tutti)

   · `plot_title::String`: titolo grande del grafico

   · `plot_size`: una Pair che indica la dimensione complessiva del grafico
  =#
  subplots = []
  distinct_parameters, _ = categorise_parameters(parameter_super)
  #
  # Calcola il minimo e il massimo valore delle ordinate tra tutti i dati,
  # per poter impostare una scala universale che mostri tutti i grafici
  # nello stesso modo (e non tagli fuori nulla).
  # In pratica: `extrema` calcola gli estremi di un Array multidimensionale, ma
  # i miei data_super sono invece Vector di Vector (di Vector, talvolta) quindi non
  # si può fare semplicemente: calcolo prima la lista degli estremi di ciascun
  # `data`, poi prendo gli estremi di tutto.
  if data_super[1][1] isa Vector
    # Alcuni data_super sono Vector³, altri Vector², e nel primo caso c'è
    # un ulteriore livello da sbrogliare.
    y_minima = [minimum([minimum(data₂) for data₂ in data₁])
                for data₁ in data_super]
    y_maxima = [maximum([maximum(data₂) for data₂ in data₁])
                for data₁ in data_super]
  else
    y_minima = [minimum(data) for data in data_super]
    y_maxima = [maximum(data) for data in data_super]
  end
  ylimits = (minimum(y_minima), maximum(y_maxima))
  #
  for (p, data) in zip(parameter_super, data_super)
    time_step_list = construct_step_list(p)
    skip_steps = p["skip_steps"]
    time_step_list_filtered = time_step_list[1:skip_steps:end]
    dataᵀ = hcat(data...)
    N = size(dataᵀ, 1)
    if displayed_sites == nothing
      displayed_sites = 1:N
    end
    #
    this_plot = plot()
    for (j, lab, lst) in zip(displayed_sites, labels, linestyles)
      plot!(this_plot, time_step_list_filtered,
                       dataᵀ[j,:],
                       ylim=ylimits,
                       label=lab,
                       linestyle=lst,
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
    shared_title_fake_plot(plot_title, parameter_super),
    Plots.plot(subplots...,
               layout=(Int(ceil(length(subplots)/2)), 2),
               size=plot_size),
    layout=grid(2, 1, heights=[0.1, 0.9])
  )
  return group_plot
end
#
function plot_standalone(data_super, parameter_super; labels, linestyles, x_label, y_label, plot_title, plot_size)
  #= Disegna un grafico dei dati di un array "_super" che non dipendono dal
     tempo, ottenuti dalle simulazioni associate ai vari insiemi di parametri
     in `parameter_super`, raggruppando in un unica immagine i grafici
     relativi alla stessa grandezza.

     Argomenti
     ---------
   · `data_super::Array{Any}`: un array di dimensione uguale a quella di
     parameter_lists.
     Ciascun elemento di data_super è un array che contiene dei dati
     da graficare, eventualmente un insieme di array da rappresentare
     contemporaneamente.

   · `labels::Array{String}`: un array di dimensione M che contiene le
     etichette da assegnare alla linea di ciascuna quantità da disegnare.

   · `linestyles::Array{Symbol}`: come `labels`, ma per gli stili delle linee.

   · `input_xlabel::String`: etichetta delle ascisse (comune a tutti)

   · `input_ylabel::String`: etichetta delle ordinate (comune a tutti)

   · `plot_title::String`: titolo grande del grafico

   · `plot_size`: una Pair che indica la dimensione complessiva del grafico
  =#
  subplots = []
  distinct_parameters, _ = categorise_parameters(parameter_super)
  #
  # Calcola il minimo e il massimo valore delle ordinate tra tutti i dati,
  # per poter impostare una scala universale che mostri tutti i grafici
  # nello stesso modo (e non tagli fuori nulla).
  # In pratica: `extrema` calcola gli estremi di un Array multidimensionale, ma
  # i miei data_super sono invece Vector di Vector (di Vector, talvolta) quindi non
  # si può fare semplicemente: calcolo prima la lista degli estremi di ciascun
  # `data`, poi prendo gli estremi di tutto.
  if data_super[1][1] isa Vector
    # Alcuni data_super sono Vector³, altri Vector², e nel primo caso c'è
    # un ulteriore livello da sbrogliare.
    y_minima = [minimum([minimum(data₂) for data₂ in data₁])
                for data₁ in data_super]
    y_maxima = [maximum([maximum(data₂) for data₂ in data₁])
                for data₁ in data_super]
  else
    y_minima = [minimum(data) for data in data_super]
    y_maxima = [maximum(data) for data in data_super]
  end
  ylimits = (minimum(y_minima), maximum(y_maxima))
  #
  for (p, data) in zip(parameter_super, data_super)
    this_plot = plot()
    for (j, (lab, lst)) in enumerate(zip(labels, linestyles))
      plot!(this_plot, data[:,j],
                       ylim=ylimits,
                       label=lab,
                       linestyle=lst,
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
    shared_title_fake_plot(plot_title, parameter_super),
    Plots.plot(subplots...,
               layout=(Int(ceil(length(subplots)/2)), 2),
               size=plot_size),
    layout=grid(2, 1, heights=[0.1, 0.9])
  )
  return group_plot
end

