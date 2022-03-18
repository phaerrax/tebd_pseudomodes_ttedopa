#!/usr/bin/julia

using LaTeXStrings
using ProgressMeter
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

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")

# Questo programma calcola l'evoluzione della catena di spin isolata,
# usando le tecniche dei MPS ed MPO.

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
  current_super = []
  timesteps_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
    # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
    H = (diagm(-1 => repeat([1], n_sites-1)) +
         diagm(1 => repeat([1], n_sites-1)))
    J = [diagm(1 => [l == k ? 0.5im : 0 for l ∈ 1:n_sites-1]) +
         diagm(-1 => [l == k ? -0.5im : 0 for l ∈ 1:n_sites-1])
         for k ∈ 1:n_sites-1]
    U = exp(-im * time_step * H)

    # Simulazione
    # ===========
    # Determina lo stato iniziale a partire dalla stringa data nei parametri
    ψ = zeros(ComplexF64, n_sites)
    ψ[1] = 1.0

    occn(ϕ) = abs2.(ϕ)
    current(ϕ) = real.(dot.(Ref(ϕ), J, Ref(ϕ)))

    # Misuro le osservabili sullo stato iniziale
    occn_list = [occn(ψ)]
    current_list = [current(ψ)]

    for _ in time_step_list[2:end]
      ψ = U * ψ
      push!(occn_list, occn(ψ))
      push!(current_list, current(ψ))
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occn_list = mapreduce(permutedims, vcat, occn_list)
    current_list = mapreduce(permutedims, vcat, current_list)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => time_step_list)
    for (j, name) in enumerate([Symbol("occ_n$n") for n = 1:n_sites])
      push!(dict, name => occn_list[:,j])
    end
    for (j, name) in enumerate([Symbol("current$n") for n = 1:n_sites-1])
      push!(dict, name => current_list[:,j])
    end
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    push!(timesteps_super, time_step_list)
    push!(occ_n_super, occn_list)
    push!(current_super, current_list)
  end

  #= Grafici
     =======
     Come funziona: creo un grafico per ogni tipo di osservabile misurata. In
     ogni grafico, metto nel titolo tutti i parametri usati, evidenziando con
     la grandezza del font o con il colore quelli che cambiano da una
     simulazione all'altra.
  =#
  plotsize = (600, 400)

  # Grafico dei numeri di occupazione (tutti i siti)
  # ------------------------------------------------
  N = size(occ_n_super[begin], 2)
  plt = groupplot(timesteps_super,
                  occ_n_super,
                  parameter_lists;
                  labels=hcat(string.(1:N)...),
                  linestyles=hcat(repeat([:solid], N...)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle n_i(t)\rangle",
                  plottitle="Numeri di occupazione",
                  plotsize=plotsize)
                  
  savefig(plt, "occ_n.png")

  N = size(current_super[begin], 2)
  plt = groupplot(timesteps_super,
                  current_super,
                  parameter_lists;
                  labels=hcat(string.(1:N)...),
                  linestyles=hcat(repeat([:solid], N...)),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\langle j_{k,k+1}(t)\rangle",
                  plottitle="Corrente tra spin adiacenti",
                  plotsize=plotsize)
                  
  savefig(plt, "current.png")


  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
