#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

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
  timesteps_super = []
  occn_abserr_super = []
  occn_relerr_super = []
  range_spins_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Leggo dai file temporanei i dati prodotti dagli script.
    tmpfilename = replace(parameters["filename"], ".json" => ".PM.dat")
    PMdata = DataFrame(CSV.File(tmpfilename))

    time_PM = PMdata[:, :time]
    norm_PM = PMdata[:, :norm_PM]
    range_spins = sort([parse(Int, split(str, "_S")[end])
                        for str ∈ names(PMdata, r"occn_PM_S")])
    occnlist = [PMdata[:, Symbol("occn_PM_S$n")]
                for n ∈ eachindex(range_spins)]
    occn_PM = reduce(hcat, occnlist)

    tmpfilename = replace(parameters["filename"], ".json" => ".TTEDOPA.dat")
    TTEDOPAdata = DataFrame(CSV.File(tmpfilename))

    time_TTEDOPA = TTEDOPAdata[:, :time_TTEDOPA]
    norm_TTEDOPA = TTEDOPAdata[:, :norm_TTEDOPA]
    occnlist = [TTEDOPAdata[:, Symbol("occn_TTEDOPA_S$n")]
                for n ∈ eachindex(range_spins)]
    occn_TTEDOPA = reduce(hcat, occnlist)

    # Creo una tabella con i dati da scrivere nel file di output
    outfilename = replace(parameters["filename"], ".json" => ".dat")
    @assert time_TTEDOPA == time_PM
    dict = Dict(:time => time_PM)
    abserr = occn_TTEDOPA .- occn_PM
    for n ∈ eachindex(range_spins)
      push!(dict, Symbol("occn_TTEDOPA_S$n") => occn_TTEDOPA[:, n])
      push!(dict, Symbol("occn_PM_S$n")      => occn_PM[:, n])
      push!(dict, Symbol("occn_abserr_S$n")  => abserr[:, n])
    end
    push!(dict, :norm_TTEDOPA => norm_TTEDOPA)
    push!(dict, :norm_PM => norm_PM)
    table = DataFrame(dict)
    CSV.write(outfilename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_PM)
    push!(occn_abserr_super, abserr)
    push!(occn_relerr_super, abserr ./ occn_TTEDOPA)
    push!(range_spins_super, range_spins)
  end

  # Plots
  # -----
  @info "Drawing plots."

  # Common options for group plots
  nrows = Int(ceil(tot_sim_n / 2))
  group_opts = @pgf {
    group_style = {
      group_size        = "$nrows by 2",
      y_descriptions_at = "edge left",
      horizontal_sep    = "2cm",
      vertical_sep      = "2cm"
    },
    no_markers,
    grid       = "major",
    legend_pos = "outer north east"
  }

  # Occupation numbers, absolute error btw. TTEDOPA and pseudomodes
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle_\rm{TTEDOPA}-\langle n_i(t)\rangle_\rm{PM}",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           occn_abserr_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.("S", 1:N) ))
      push!(grp, ax)
    end
    pgfsave("population_absolute_error.pdf", grp)
  end

  # Occupation numbers, relative error btw. TTEDOPA and pseudomodes
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"1-\langle n_i(t)\rangle_\rm{TTEDOPA}/\langle n_i(t)\rangle_\rm{PM})",
        ymin = -1,
        ymax = 1
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           occn_abserr_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.("S", 1:N) ))
      push!(grp, ax)
    end
    pgfsave("population_relative_error.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
