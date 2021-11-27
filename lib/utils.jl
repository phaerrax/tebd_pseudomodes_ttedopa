using Plots
using Measures
using LinearAlgebra
using JSON
using Base.Filesystem
using ITensors

#= Questo file contiene un insieme variegato di funzioni che servono bene o
   male per tutti gli script. Le raduno tutte qui in modo da non dover copiare
   il loro codice ad ogni nuovo file.
=#

# Lettura dei parametri della simulazione
# =======================================
function isjson(filename::String)
  return length(filename) > 5 && filename[end-4:end] == ".json"
  # Volutamente, un file che di nome fa solo ".json" viene ignorato.
end

function load_parameters(file_list)
  if isempty(file_list)
    throw(ErrorException("Non è stato fornito alcun file di parametri."))
  end
  first_arg = file_list[1]
  prev_dir = pwd()
  if isdir(first_arg)
    # Se il primo input è una cartella, legge tutti i file .json al suo
    # interno e li carica come file di parametri; alla fine i file di
    # output saranno salvati in tale cartella.
    # I restanti elementi di ARGS vengono ignorati (l'utente viene avvisato).
    cd(first_arg)
    files = filter(isjson, readdir())
    @info "$first_arg è una cartella. I restanti argomenti passati alla "*
          "linea di comando saranno ignorati."
  else
    # Se il primo input non è una cartella, tutti gli argomenti passati in
    # ARGS vengono trattati come file da leggere contenenti i parametri.
    # I file di output saranno salvati nella cartella pwd() al momento
    # del lancio del programma.
    files = file_list
  end
  # Carico i file di parametri nei dizionari.
  parameter_lists = []
  for f in files
    open(f) do input
      s = read(input, String)
      # Aggiungo anche il nome del file alla lista di parametri.
      push!(parameter_lists, merge(Dict("filename" => f), JSON.parse(s)))
    end
  end
  cd(prev_dir)
  return parameter_lists
end

# Costruzione della lista di istanti di tempo per la simulazione
# ==============================================================
function construct_step_list(parameters)
  τ = parameters["simulation_time_step"]
  end_time = parameters["simulation_end_time"]
  return collect(range(0, end_time; step=τ))
end

# Calcolo di osservabili durante la simulazione
# =============================================
# Per calcolare gli autostati dell'operatore numero: calcolo anche la somma
# di tutti i coefficienti (così da verificare che sia pari a 1).
function levels(projs::Vector{MPO}, state::MPS, ::SiteType"S=1/2")
  lev = [real(inner(state, p * state)) for p in projs]
  return [lev; sum(lev)]
end
function levels(projs::Vector{MPS}, state::MPS)
  lev = [real(inner(p, state)) for p in projs]
  return [lev; sum(lev)]
end

# Rilevazione della dimensione dei legami tra i siti degli MPS o MPO
function linkdims(m::MPS)
  return [ITensors.dim(linkind(m, j)) for j ∈ 1:length(m)-1]
end
function linkdims(m::MPO)
  return [ITensors.dim(linkind(m, j)) for j ∈ 1:length(m)-1]
end

# Composizione di MPS e MPO
# =========================
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

function chain(left::MPO, right::MPO)
  # Come sopra, ma per due MPO.
  midN = length(left) # È l'indice a cui si troverà il "buco"
  #for j in eachindex(left)
  #  replacetags!(left[j], "l=$j", "l=$j"; tags="Link")
  #  replacetags!(left[j], "l=$(j-1)", "l=$(j-1)"; tags="Link")
  #end
  for j in eachindex(right)
    replacetags!(right[j], "l=$j", "l=$(midN+j)"; tags="Link")
    replacetags!(right[j], "l=$(j-1)", "l=$(midN+j-1)"; tags="Link")
  end
  M = MPO([left..., right...])
  missing_link = Index(1; tags="Link,l=$midN")
  M[midN] = M[midN] * state(missing_link, 1)
  M[midN+1] = M[midN+1] * state(dag(missing_link), 1)
  # L'ordine degli Index negli ITensor M[midN] e M[midN+1] non è quello
  # che si otterrebbe costruendo normalmente un MPO che si estende su
  # tutto l'array dei siti; per la precisione i due Index di tipo "Link"
  # sono scambiati.
  # Siccome però l'ordine degli Index posseduti da un ITensor non è
  # importante, conta solo quali indici ha, il risultato non dovrebbe
  # cambiare.
  return M
end

# Variante a numero di argomenti libero
chain(a::MPS, b...) = chain(a, chain(b...))
chain(a::MPO, b...) = chain(a, chain(b...))

function embed_slice(sites::Array{Index{Int64}},
                     range::UnitRange{Int},
                     slice::MPO)
  #=
  Prende un MPO definito solo su una fetta dei siti e lo estende con operatori
  identità per creare un MPO definito su tutti i siti.
  È richiesto che per i SiteType degli elementi di sites non compresi nella
  fetta data sia definito un operatore di nome "id".

  Argomenti
  ---------
  · `sites::Array{Index{Int64}}`: l'array di siti dell'intero sistema.

  · `range::UnitRange{Int}`: un intervallo che indica il sito iniziale e il
    sito finale del MPO fornito che deve essere esteso.

  · `slice::Array{ITensors}`: il MPO da estendere.
  =#
  # Controllo dei parametri
  if length(slice) != length(range)
    throw(DimensionMismatch("Le dimensioni di slice e range non combaciano."))
  end
  if !issubset(range, eachindex(sites))
    throw(BoundsError(range, sites))
  end

  if range[begin] == 1 && range[end] == length(sites)
    mpo = slice
  elseif range[begin] == 1
    mpo = chain(slice,
                MPO(sites[range[end]+1 : end], "id"))
  elseif range[end] == length(sites)
    mpo = chain(MPO(sites[1 : range[begin]-1], "id"),
                slice)
  else
    mpo = chain(MPO(sites[1 : range[begin]-1], "id"),
                slice,
                MPO(sites[range[end]+1 : end], "id"))
  end
  return mpo
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
  #= Disegna un grafico a partire dagli array "_super" ottenuti dalla
     simulazione qui sopra.

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

# Costruzione dell'operatore di evoluzione temporale
# ==================================================
# Richiede l'esistenza di due FUNZIONI links_odd e links_even che abbiano
# come argomento un singolo numero reale, che rappresenta il passo di
# evoluzione temporale.
function evolution_operator(links_odd, links_even, τ, order)
  if order == 1
    list = [links_odd(τ);
            links_even(τ)]
  elseif order == 2
    list = [links_odd(0.5τ);
            links_even(τ);
            links_odd(0.5τ)]
  elseif order == 4
    c = (2 - 2^(1/3))^(-1)
    list = [links_even(0.5c * τ);
            links_odd(c * τ);
            links_even(0.5 * (1-c) * τ);
            links_odd((1 - 2c) * τ);
            links_even(0.5 * (1-c) * τ);
            links_odd(c * τ);
            links_even(0.5c * τ)]
  else
    throw(DomainError(order,
                      "L'espansione di Trotter-Suzuki all'ordine $order "*
                      "non è supportata. Attualmente sono disponibili "*
                      "solo le espansioni con ordine 1, 2 o 4."))
  end
  return list
end
