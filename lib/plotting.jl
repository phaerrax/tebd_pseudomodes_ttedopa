using Measures
using Base.Filesystem
using Plots
using JSON

"""
    categorise_parameters(parameter_lists)

Parse the lists in `parameter_lists` and splits them in two list according
whether they are unique or repeated.

Returns two lists, `unique` and `repeated`, packed into a tuple.
"""
function categorise_parameters(parameter_lists)
  distinct = String[]
  # We need to examine each parameter list to check if the dimension of the
  # oscillator Hilbert spaces are given separately for the cold and hot baths,
  # or if there's one dimension given for both.
  # In the latter case, we duplicate the information to make it like the
  # former.
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
  # Load the dictionary to assign a symbol to each parameter, to be displayed
  # in the plots.
  short_name = JSON.parse(read(f, String))
  close(f)
  hidden_parameters = ["skip_steps", "filename"]
  return join([short_name[k] * "=" * string(values_dict[k])
               for k in setdiff(keys, hidden_parameters)],
              ", ")
end

"""
    shared_title_fake_plot(subject::String, parameters; filenameastitle = false)
    
Create a fake plot containing only a string, which acts as the title of a
group of other subplots.

Since Plots.jl does not yet provide a mean to assign a title to a plot group,
we make do with this function.
The title will contain the parameters which are common to all subplots.
"""
function shared_title_fake_plot(subject::String,
                                parameters;
                                filenameastitle = false)
  # The big title contains parameters common to all simulations, therefore we
  # can choose any element from `parameter_lists` and the result is the same.
  
  if filenameastitle
    bigtitle = subject
  else
    # Parameters we don't want to appear in the big title:
    hidden_parameters = ["simulation_end_time", "skip_steps", "filename"]
    _, repeated_parameters = categorise_parameters(parameters)
    #
    rootdirname = "simulazioni_tesi"
    sourcepath = Base.source_path()
    ind = findfirst(rootdirname, sourcepath)
    rootpath = sourcepath[begin:ind[end]]
    libpath = joinpath(rootpath, "lib")
    f = open(joinpath(libpath, "short_names_dictionary.json"), "r")
    # Load the dictionary to assign a symbol to each parameter, to be displayed
    # in the plots.
    short_name = JSON.parse(read(f, String))
    close(f)
    #
    shared_title = join([short_name[k] * "=" * string(parameters[begin][k])
                         for k in setdiff(repeated_parameters, hidden_parameters)],
                        ", ")
    bigtitle = subject * "\n" * shared_title
  end
  y = ones(3) # Fake data in order to create an empty plot
  title_fake_plot = Plots.scatter(y, marker=0, markeralpha=0, ticks=nothing, annotations=(2, y[2], Plots.text(bigtitle)), axis=false, grid=false, leg=false, bottom_margin=2cm, size=(200,100))
  return title_fake_plot
end

"""
    groupplot(x_super, y_super, parameter_super; maxyrange = nothing, rescale = true, filenameastitle = false, labels, linestyles, commonxlabel, commonylabel, plottitle, plotsize)

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
  the filename of the JSON file describing the parameters of the simulation
- `plottitle` → title of the plot group.
- `plotsize` → a Pair denoting the size of the individual subplots.

# Usage
Each element ``Yᵢ ∈`` `y_super` is what we would pass to `Plots.plot`, together
with the respective element ``Xᵢ ∈`` `x_super`, to draw its plot.
In other words, each ``Yᵢ ∈`` `y_super` is a `M×N` matrix, with
`M = length(Xᵢ)`; `N` is the number of columns, each of which is a separate
line in the plot.
For example, in order to draw the occupation numbers of 10 sites all
together in a single plot, we create a 10-column matrix, with the n-th
column containing the occupation numbers of the n-th site at each instant.
"""
function groupplot(x_super,
    y_super,
    parameter_super;
    maxyrange = nothing,
    rescale = true,
    filenameastitle = true,
    labels,
    linestyles,
    commonxlabel,
    commonylabel,
    plottitle,
    plotsize)
  # If `labels` is a vector of row vectors of strings, then each subplot
  # already has its own set of labels (each column of the row vector corresponds
  # to the columns in each element of `y_super`).
  # If `labels` is only a row vector of strings instead, then this means that
  # the labels are common to all plots, so we repeat them creating a vector
  # of identical copies.
  if labels isa Matrix{String}
    newlabels = repeat([labels], length(parameter_super))
    labels = newlabels
  end
  # Now we do the same, but with `linestyles`...
  if linestyles isa Matrix{Symbol}
    newlinestyles = repeat([linestyles], length(parameter_super))
    linestyles = newlinestyles
  end
  # ...and `maxyrange`.
  if !isnothing(maxyrange) && !isa(maxyrange, Vector)
    newmaxyrange = repeat([maxyrange], length(parameter_super))
    maxyrange = newmaxyrange
  end

  if rescale
    # We set a common scale for the y-axis among all subplots: this makes
    # comparing the results easier.
    yminima = minimum.(y_super)
    ymaxima = maximum.(y_super)
    ylimits = (minimum(yminima), maximum(ymaxima))
    ylimits_super = repeat([ylimits], length(y_super))
  else
    ylimits_super = extrema.(y_super)
  end

  # We need to limit the y-axis to `maxyrange` 𝑜𝑛𝑙𝑦 if the range isn't already
  # within it.
  # TODO: try to simplify the code here by using `clamp` (if only I knew how
  # to make it work with tuples...).
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

  # Sort the parameters into repeated and not-repeated, for the plot titles.
  if filenameastitle
    subplottitles = [p["filename"] for p ∈ parameter_super]
  else
    distinct_parameters, _ = categorise_parameters(parameter_super)
    subplottitles = [subplot_title(p, distinct_parameters)
                     for p ∈ parameter_super]
  end

  # Compute the dimension of the whole image, starting from the size of the
  # single subplot.
  figuresize = (2, Int(ceil(length(x_super)/2))+0.5) .* plotsize

  # Create the subplots.
  subplots = [Plots.plot(X,
                         Y,
                         ylim=ylimits,
                         label=lab,
                         linestyle=lst,
                         legend=:outerright,
                         xlabel=commonxlabel,
                         ylabel=commonylabel,
                         title=sptitle,
                         left_margin=5mm,
                         bottom_margin=5mm,
                         size=figuresize)
              for (X, Y, lab, lst, ylimits, sptitle, p) ∈ zip(x_super,
                                                              y_super,
                                                              labels,
                                                              linestyles,
                                                              ylimits_super,
                                                              subplottitles,
                                                              parameter_super)]
  # The subplots will be arranged into a two-column grid; if we have an odd
  # number of plots, we need to create an empty one in order to fill the gap.
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
  # Create the bigger plot which will contain all subplots, and the title.
  group = Plots.plot(shared_title_fake_plot(plottitle,
                                            parameter_super;
                                            filenameastitle),
                     Plots.plot(subplots..., layout=(length(subplots)÷2, 2)),
                     # Use ÷, not /, to get an Int
                     layout=Plots.grid(2, 1, heights=[0.1, 0.9]))
  return group
end

"""
    unifiedplot(x_super, y_super, parameter_super; filenameastitle = false, linestyle, xlabel, ylabel, plottitle, plotsize)

Create a plot where all data series (X,Y) in `x_super` × `y_super` are drawn
together. A title is shown on top, which lists the common parameters among the
subplots; parameters which differ among the plots are instead displayed
in the label of each series.

# Arguments
- `x_super` → list of lists, each one representing the x-axis of a subplot..
- `y_super` → list of lists, corresponding to the relative element in `x_super`.
- `parameter_lists` → list of dictionaries, containing the parameters
  associated to each element in `x_super`.
- `filenameastitle` → if `true`, the label of each subplot will be given by
  the filename of the JSON file describing the parameters of the simulation
- `linestyle` → line styles of the plots
- `xlabel` → x-axis label.
- `ylabel` → y-axis label.
- `plottitle` → title of the plot.
- `plotsize` → a Pair representing the plot size (in pixels).
"""
function unifiedplot(x_super, y_super, parameter_super; filenameastitle = false, linestyle, xlabel, ylabel, plottitle, plotsize)
  # Sort the parameters into repeated and not-repeated, for the plot titles.
  if filenameastitle
    labels = [p["filename"] for p ∈ parameter_super]
  else
    distinct_parameters, _ = categorise_parameters(parameter_super)
    labels = [subplot_title(p, distinct_parameters)
                     for p ∈ parameter_super]
  end
  # Set the figure size (allowing a little more space for the title).
  figuresize = (1, 1.25) .* plotsize
  plt = Plots.plot()
  for (X, Y, lb, p) in zip(x_super, y_super, labels, parameter_super)
    plt = Plots.plot!(X,
                      Y,
                      label=lb,
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
    unifiedlogplot(x_super, y_super, parameter_super; <keyword arguments>)

Create a plot where all data series (X,Y) in `x_super` × `y_super` are drawn
together, with the y-axis in logarithmic scale.
See `unifiedplot` for more details on the keyword arguments.

Note that no check is performed whether the arguments of the logarithm are
positive.
"""
function unifiedlogplot(x_super, y_super, parameter_super; kwargs...)
  return unifiedplot(x_super,
                     [log.(Y) for Y ∈ y_super],
                     parameter_super;
                     kwargs...)
end
