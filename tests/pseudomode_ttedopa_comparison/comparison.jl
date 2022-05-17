#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV

# Se lo script viene eseguito su Qtech, devo disabilitare l'output
# grafico altrimenti il programma si schianta.
if gethostname() == "qtech.fisica.unimi.it" ||
   gethostname() == "qtech2.fisica.unimi.it"
  ENV["GKSwstype"] = "100"
  @info "Esecuzione su server remoto. Output grafico disattivato."
else
  delete!(ENV, "GKSwstype")
  # Se la chiave "GKSwstype" non esiste non succede niente.
end

rootdirname = "simulazioni_tesi"
sourcepath = Base.source_path()
# Cartella base: determina il percorso assoluto del file in esecuzione, e
# rimuovi tutto ciò che segue rootdirname.
ind = findfirst(rootdirname, sourcepath)
rootpath = sourcepath[begin:ind[end]]
# `rootpath` è la cartella principale del progetto.
libpath = joinpath(rootpath, "lib")

include(joinpath(libpath, "utils.jl"))
include(joinpath(libpath, "plotting.jl"))
include(joinpath(libpath, "spin_chain_space.jl"))
include(joinpath(libpath, "harmonic_oscillator_space.jl"))
include(joinpath(libpath, "operators.jl"))
include(joinpath(libpath, "tedopa.jl"))

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
    tmpfilename = replace(parameters["filename"], ".json" => ".dat~1")
    PMdata = DataFrame(CSV.File(tmpfilename))

    time_PM = PMdata[:, :time]
    norm_PM = PMdata[:, :norm_PM]
    range_spins = sort([parse(Int, split(str, "_S")[end])
                        for str ∈ names(PMdata, r"occn_PM_S")])
    occnlist = [PMdata[:, Symbol("occn_PM_S$n")]
                for n ∈ eachindex(range_spins)]
    occn_PM = reduce(hcat, occnlist)

    tmpfilename = replace(parameters["filename"], ".json" => ".dat~2")
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

  # Grafici
  # =======
  plotsize = (600, 400)

  # Grafico dei numeri di occupazione (solo spin)
  # ---------------------------------------------
  plt = groupplot(timesteps_super,
                  occn_abserr_super,
                  parameter_lists;
                  labels=[reduce(hcat, ["S$n" for n ∈ eachindex(rn)])
                          for rn ∈ range_spins_super],
                  linestyles=[reduce(hcat, repeat([:solid], length(rn)))
                              for rn ∈ range_spins_super],
                  commonxlabel=L"t",
                  commonylabel=L"\langle n_i(t)\rangle_\rm{TTEDOPA}-\langle n_i(t)\rangle_\rm{PM}",
                  plottitle="Differenza numeri di occupazione spin (TTEDOPA - p.modi)",
                  plotsize=plotsize)
  savefig(plt, "occn_abserr.png")

  plt = groupplot(timesteps_super,
                  occn_relerr_super,
                  parameter_lists;
                  maxyrange=(-1,1),
                  labels=[reduce(hcat, ["S$n" for n ∈ eachindex(rn)])
                          for rn ∈ range_spins_super],
                  linestyles=[reduce(hcat, repeat([:solid], length(rn)))
                              for rn ∈ range_spins_super],
                  commonxlabel=L"t",
                  commonylabel=L"1-\langle n_i(t)\rangle_\rm{TTEDOPA}/\langle n_i(t)\rangle_\rm{PM})",
                  plottitle="Err. relativo numeri di occupazione spin (TTEDOPA - p.modi)/TTEDOPA",
                  plotsize=plotsize)
  savefig(plt, "occn_relerr.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
