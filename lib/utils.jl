using Plots
using Measures
using LinearAlgebra
using JSON
using Base.Filesystem

#= Questo file contiene un insieme variegato di funzioni che servono bene o
   male per tutti gli script. Le raduno tutte qui in modo da non dover copiare
   il loro codice ad ogni nuovo file.
=#

# Lettura dei parametri della simulazione
# =======================================
function load_parameters(file_list)
  first_arg = file_list[1]
  prev_dir = pwd()
  if isdir(first_arg)
    # Se il primo input è una cartella, legge tutti i file .json al suo interno
    # e li carica come file di parametri; alla fine i file di output saranno
    # salvati in tale cartella.
    # I restanti elementi di ARGS vengono ignorati (l'utente viene avvisato).
    cd(first_arg)
    files = filter(s -> s[end-4:end] == ".json", readdir())
    @info "$first_arg è una cartella. I restanti argomenti passati alla linea
    di comando saranno ignorati."
  else
    # Se il primo input non è una cartella, tutti gli argomenti passati in ARGS
    # vengono trattati come file da leggere contenenti i parametri.
    # I file di output saranno salvati nella cartella pwd() al momento del lancio
    # del programma.
    files = file_list
  end
  # Carico i file di parametri nei dizionari.
  parameter_lists = []
  for f in files
    open(f) do input
      s = read(input, String)
      push!(parameter_lists, JSON.parse(s))
    end
  end
  cd(prev_dir)
  return parameter_lists
end

# Costruzione della lista di istanti di tempo per la simulazione
# ==============================================================
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

# Grafici
# =======
function categorise_parameters(parameter_lists)
  # Analizzo l'array dei parametri e individuo quali parametri variano tra un
  # caso e l'altro, suddividendoli tra "distinti" se almeno uno dei parametri
  # è diverso tra le simulazioni, e "ripetuti" se sono invece tutti uguali.
  distinct = String[]
  for key in keys(parameter_lists[begin])
    test_list = [p[key] for p in parameter_lists]
    if !all(x -> x == first(test_list), test_list)
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
  return join(
    [short_name[k] * "=" * string(values_dict[k]) for k in keys],
    ", "
   )
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
  hidden_parameters = ["simulation_end_time"]
  _, repeated_parameters = categorise_parameters(parameters)
  #
  f = open(lib_path * "/short_names_dictionary.json", "r")
  # Carico il dizionario dei "nomi brevi" per i parametri.
  short_name = JSON.parse(read(f, String))
  close(f)
  #
  shared_title = join(
    [short_name[k] * "=" * string(parameters[begin][k]) for k in setdiff(repeated_parameters, hidden_parameters)],
    ", "
  )
  y = ones(3) # Dati falsi per far apparire un grafico
  title_fake_plot = Plots.scatter(y, marker=0, markeralpha=0, ticks=nothing, annotations=(2, y[2], text(subject * "\n" * shared_title)), axis=false, grid=false, leg=false, bottom_margin=2cm, size=(200,100))
  return title_fake_plot
end
#
function plot_time_series(data_super, parameter_super; displayed_sites, labels, linestyles, x_label, y_label, plot_title, plot_size)
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
  =#
  subplots = []
  distinct_parameters, _ = categorise_parameters(parameter_super)
  for (p, data) in zip(parameter_super, data_super)
    time_step_list = construct_step_list(p["simulation_end_time"], p["spin_excitation_energy"])
    dataᵀ = hcat(data...)
    N = size(dataᵀ, 1)
    if displayed_sites == nothing
      displayed_sites = 1:N
    end
    #
    this_plot = plot()
    for (j, lab, lst) in zip(displayed_sites, labels, linestyles)
      plot!(this_plot, time_step_list,
                       dataᵀ[j,:],
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
    Plots.plot(subplots..., layout=length(subplots), size=plot_size),
    layout=grid(2, 1, heights=[0.1, 0.9])
  )
  return group_plot
end

function chain(left::MPS, right::MPS)
  # Questa funzione dovrebbe essere l'analogo di `chain` di mpnum: prende
  # due MPS e ne crea uno che ne è la concatenazione.
  # Per scriverla mi sono "ispirato" al codice della funzione `MPS` nel
  # file mps.jl, riga 308.

  midN = length(left) # È l'indice a cui si troverà il "buco"
  # Innanzitutto, riscalo i tag dei Link negli MPS di argomento, in
  # modo che la numerazione alla fine risulti corretta.
  # Facendo un po' di prove ho visto che la numerazione degli Index di
  # tipo Link nei MPS segue uno schema proprio, cioè non è legata a come
  # vengono chiamati i siti: va semplicemente da 1 alla lunghezza-1.
  for j in eachindex(left)
    replacetags!(left[j], "l=$j", "l=$j"; tags="Link")
    replacetags!(left[j], "l=$(j-1)", "l=$(j-1)"; tags="Link")
  end
  for j in eachindex(right)
    replacetags!(right[j], "l=$j", "l=$(midN+j)"; tags="Link")
    replacetags!(right[j], "l=$(j-1)", "l=$(midN+j-1)"; tags="Link")
  end
  # Spacchetto i tensori in `left` e `right` e creo un MPS con le
  # liste concatenate.
  M = MPS([left..., right...])
  # I siti "di confine", l'ultimo a dx di `left` e il primo a sx di
  # `right`, non sono ancora collegati. Creo un indice banale, di
  # dimensione 1, di tipo Link e lo aggiungo a quei due siti.
  # L'indice è _sempre_ banale perché per costruzione il MPS risultato
  # è il prodotto tensore dei due argomenti: non c'è interazione tra loro.
  missing_link = Index(1; tags="Link,l=$midN")
  M[midN] = M[midN] * state(missing_link, 1)
  M[midN+1] = state(dag(missing_link), 1) * M[midN+1]

  return M
end

# Variante a numero di argomenti libero
chain(a::MPS, b...) = chain(a, chain(b...))
