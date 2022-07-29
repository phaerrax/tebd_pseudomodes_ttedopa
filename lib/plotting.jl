using Measures
using Base.Filesystem
using Plots
using JSON

# Grafici
# =======
"""
    categorise_parameters(parameter_lists)

Parse the lists in `parameter_lists` and splits them in two list according
whether they are unique or repeated.

Returns two lists, `unique` and `repeated`, packed into a tuple.
"""
function categorise_parameters(parameter_lists)
  distinct = String[]

  # Devo controllare ciascuna lista di parametri per controllare se specifica
  # le dimensioni degli oscillatori separatamente per quelli caldi e freddi
  # o se dà solo una dimensione per entrambi.
  # Nel secondo caso, modifico il dizionario per riportarlo al primo caso.
  for dict in parameter_lists
    if (haskey(dict, "oscillator_space_dimension") &&
        !haskey(dict, "hot_oscillator_space_dimension") &&
        !haskey(dict, "cold_oscillator_space_dimension"))
      hotoscdim = pop!(dict, "oscillator_space_dimension")
      coldoscdim = hotoscdim
    end
  end

  for key in keys(parameter_lists[begin])
    test_list = [p[key] for p in parameter_lists]
    if !allequal(test_list)
      push!(distinct, key)
    end
  end
  repeated = setdiff(keys(parameter_lists[begin]), distinct)
  return distinct, repeated
end

"""
    subplot_title(values_dict, keys)

Create a custom title for each subplot in a group, displaying the parameters
which have different values among the subplots.
"""
function subplot_title(values_dict, keys)
  rootdirname = "simulazioni_tesi"
  sourcepath = Base.source_path()
  ind = findfirst(rootdirname, sourcepath)
  rootpath = sourcepath[begin:ind[end]]
  libpath = joinpath(rootpath, "lib")
  f = open(joinpath(libpath, "short_names_dictionary.json"), "r")
  # Carico il dizionario dei "nomi brevi" per i parametri.
  short_name = JSON.parse(read(f, String))
  close(f)
  hidden_parameters = ["skip_steps", "filename"]
  return join([short_name[k] * "=" * string(values_dict[k])
               for k in setdiff(keys, hidden_parameters)],
              ", ")
end

"""
    shared_title_fake_plot(subject::String, parameters)
    
Create a fake plot containing only a string, which acts as the title of a
group of other subplots.

Since Plots.jl does not yet provide a mean to assign a title to a plot group,
we make do with this function.
The title will contain the parameters which are common to all subplots.
"""
function shared_title_fake_plot(subject::String, parameters)
  # Siccome l titolone contiene i parametri comuni a tutte le simulazioni,
  # posso prendere senza problemi uno degli elementi di parameter_lists
  # qualunque) e lo uso come titolo del gruppo di grafici.
  #
  # Inserire in questo array i parametri che non si vuole che appaiano nel
  # titolo:
  hidden_parameters = ["simulation_end_time", "skip_steps", "filename"]
  _, repeated_parameters = categorise_parameters(parameters)
  #
  rootdirname = "simulazioni_tesi"
  sourcepath = Base.source_path()
  ind = findfirst(rootdirname, sourcepath)
  rootpath = sourcepath[begin:ind[end]]
  libpath = joinpath(rootpath, "lib")
  f = open(joinpath(libpath, "short_names_dictionary.json"), "r")
  # Carico il dizionario dei "nomi brevi" per i parametri.
  short_name = JSON.parse(read(f, String))
  close(f)
  #
  shared_title = join([short_name[k] * "=" * string(parameters[begin][k])
                       for k in setdiff(repeated_parameters, hidden_parameters)],
                      ", ")
  y = ones(3) # Dati falsi per far apparire un grafico
  title_fake_plot = Plots.scatter(y, marker=0, markeralpha=0, ticks=nothing, annotations=(2, y[2], Plots.text(subject * "\n" * shared_title)), axis=false, grid=false, leg=false, bottom_margin=2cm, size=(200,100))
  return title_fake_plot
end

"""
    groupplot(x_super, y_super, parameter_super; maxyrange = nothing, rescale = true, labels, linestyles, commonxlabel, commonylabel, plottitle, plotsize)

Create an image which groups the plots of each ``(X,Y) ∈ x_super × y_super``;
on top of the plots, a title is created which lists the common parameters among
the subplots; parameters which differ among the plots are instead displayed
above each subplot.

# Arguments
- `x_super` → list of lists, each one representing the x-axis of a subplot..
- `y_super` → list of lists, corresponding to the relative element in `x_super`.
  It tries to mimic the Plots.jl syntax, i.e. when the elements in `y_super`
  are matrices, they are split by columns.
- `parameter_lists` → list of dictionaries, containing the parameters
  associated to each element in `x_super`.
- `labels`::Union{Matrix{String},Vector{Matrix{String}}} → array of labels for
  the lines in each subplot. If it is a Vector, it is split among the subplots,
  else it is repeated for each one of them.
- `linestyles` → just like `labels`, but for line styles.
- `commonxlabel` → x-axis label (common to all subplots).
- `commonylabel` → y-axis label (common to all subplots).
- `maxyrange` → y-axis range at which the subplots may be clipped
- `rescale` → if `true`, all subplots are rescaled so that they have the same
  y-axis.
- `plottitle` → title of the plot group.
- `plotsize` → a Pair denoting the size of the individual subplots.
"""
function groupplot(x_super,
    y_super,
    parameter_super;
    maxyrange = nothing,
    rescale = true,
    labels,
    linestyles,
    commonxlabel,
    commonylabel,
    plottitle,
    plotsize)
  #=
  `y_super` è un array, ogni elemento del quale è quello che si passerebbe
  alla funzione Plots.plot insieme al rispettivo elemento Xᵢ ∈ x_super
  per disegnarne il grafico. Questo significa che ogni Yᵢ ∈ y_super è una 
  matrice M×N con M = length(Xᵢ); N è il numero di colonne, e ogni
  colonna rappresenta una serie di dati da graficare. Ad esempio, per
  graficare tutti assieme i numeri di occupazione di 10 siti si può
  creare una matrice di 10 colonne: ogni riga della matrice conterrà
  i numeri di occupazione dei siti a un certo istante di tempo.
  =#
  # Se `labels` è un vettore di vettori riga di stringhe, significa che
  # ogni sottografico ha già il suo insieme di etichette: sono a posto.
  # Se invece `labels` è solo un vettore riga di stringhe, significa che
  # quelle etichette sono da usare per tutti i grafici: allora creo in
  # questo momento il vettore di vettori riga di stringhe ripetendo quello
  # fornito come argomento.
  if labels isa Matrix{String}
    newlabels = repeat([labels], length(parameter_super))
    labels = newlabels
  end
  # Ripeto lo stesso trattamento per `linestyles`...
  if linestyles isa Matrix{Symbol}
    newlinestyles = repeat([linestyles], length(parameter_super))
    linestyles = newlinestyles
  end
  # ...e per `maxyrange`
  if !isnothing(maxyrange) && !isa(maxyrange, Vector)
    newmaxyrange = repeat([maxyrange], length(parameter_super))
    maxyrange = newmaxyrange
  end

  if rescale
    # Per poter meglio confrontare a vista i dati, imposto una scala delle
    # ordinate uguale per tutti i grafici.
    yminima = minimum.(y_super)
    ymaxima = maximum.(y_super)
    ylimits = (minimum(yminima), maximum(ymaxima))
    ylimits_super = repeat([ylimits], length(y_super))
  else
    ylimits_super = extrema.(y_super)
  end

  # Limito l'asse y a maxyrange 𝑠𝑜𝑙𝑜 se i grafici non ci stanno già
  # dentro da soli.
  # Forse qui si potrebbe semplificare il tutto con `clamp`, se solo
  # trovassi il modo di farla funzionare con le tuple...
  if !isnothing(maxyrange)
    for (i, maxrn) ∈ zip(eachindex(ylimits_super), maxyrange)
      if first(ylimits_super[i]) < maxrn[begin]
        ylimits_super[i] = (maxrn[begin], last(ylimits_super[i]))
      end
      if maxrn[end] < last(ylimits_super[i])
        ylimits_super[i] = (first(ylimits_super[i]), maxrn[end])
      end
    end
  end

  # Smisto i parametri in ripetuti e non, per creare i titoli dei grafici.
  distinct_parameters, _ = categorise_parameters(parameter_super)
  # Calcolo la grandezza totale dell'immagine a partire da quella dei grafici.
  figuresize = (2, Int(ceil(length(x_super)/2))+0.5) .* plotsize

  # Creo i singoli grafici.
  subplots = [Plots.plot(X,
                         Y,
                         ylim=ylimits,
                         label=lab,
                         linestyle=lst,
                         legend=:outerright,
                         xlabel=commonxlabel,
                         ylabel=commonylabel,
                         title=subplot_title(p, distinct_parameters),
                         left_margin=5mm,
                         bottom_margin=5mm,
                         size=figuresize)
              for (X, Y, lab, lst, ylimits, p) ∈ zip(x_super,
                                                     y_super,
                                                     labels,
                                                     linestyles,
                                                     ylimits_super,
                                                     parameter_super)]
  # I grafici saranno disposti in una griglia con due colonne; se ho un numero
  # dispari di grafici, ne creo uno vuoto in modo da riempire il buco che
  # si crea (altrimenti mi becco un errore).
  if isodd(length(subplots))
    fakeplot = Plots.scatter(ones(2),
                             marker=0,
                             markeralpha=0,
                             ticks=nothing,
                             axis=false,
                             grid=false,
                             leg=false,
                             size=figuresize)
    push!(subplots, fakeplot)
  end
  # Creo il grafico che raggruppa tutto, insieme al titolo principale.
  group = Plots.plot(shared_title_fake_plot(plottitle, parameter_super),
                     Plots.plot(subplots..., layout=(length(subplots)÷2, 2)),
                     # Usa ÷ e non /, in modo da ottenere un Int!
                     layout=Plots.grid(2, 1, heights=[0.1, 0.9]))
  return group
end

"""
    unifiedplot(x_super, y_super, parameter_super; linestyle, xlabel, ylabel, plottitle, plotsize)

Create a plot where all data series (X,Y) in `x_super` × `y_super` are drawn
together. A title is shown on top, which lists the common parameters among the
subplots; parameters which differ among the plots are instead displayed
in the label of each series.

# Arguments
- `x_super` → list of lists, each one representing the x-axis of a subplot..
- `y_super` → list of lists, corresponding to the relative element in `x_super`.
- `parameter_lists` → list of dictionaries, containing the parameters
  associated to each element in `x_super`.
- `linestyle` → line styles of the plots
- `xlabel` → x-axis label.
- `ylabel` → y-axis label.
- `plottitle` → title of the plot.
- `plotsize` → a Pair representing the plot size (in pixels).
"""
function unifiedplot(x_super, y_super, parameter_super; linestyle, xlabel, ylabel, plottitle, plotsize)
  # Smisto i parametri in ripetuti e non, per creare le etichette dei grafici.
  distinct_parameters, _ = categorise_parameters(parameter_super)
  # Imposto la dimensione della figura (con un po' di spazio per il titolo).
  figuresize = (1, 1.25) .* plotsize
  plt = Plots.plot()
  for (X, Y, p) in zip(x_super, y_super, parameter_super)
    plt = Plots.plot!(X,
                      Y,
                      label=subplot_title(p, distinct_parameters),
                      linestyle=linestyle,
                      legend=:outerbottom,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      title=plottitle,
                      size=figuresize)
  end
  return plt
end

"""
    unifiedlogplot(x_super, y_super, parameter_super; linestyle, xlabel, ylabel, plottitle, plotsize)

Create a plot where all data series (X,Y) in `x_super` × `y_super` are drawn
together, with the y-axis in logarithmic scale.
See `unifiedplot` for more details on the arguments.

Note that no check is performed whether the arguments of the logarithm are
positive.
"""
function unifiedlogplot(x_super, y_super, parameter_super; linestyle, xlabel, ylabel, plottitle, plotsize)
  return unifiedplot(x_super,
                     [log.(Y) for Y ∈ y_super],
                     parameter_super;
                     linestyle=linestyle,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     plottitle=plottitle,
                     plotsize=plotsize)
end
