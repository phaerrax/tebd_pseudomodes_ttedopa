#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter

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
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/operators.jl")

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
  bond_dimensions_super = []
  entropy_super = []
  current_allsites_super = []
  timesteps_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

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
    sites = siteinds("S=1/2", n_sites)

    # Costruzione dell'operatore di evoluzione
    # ========================================
    localcfs = repeat([ε], n_sites)
    interactioncfs = repeat([1], n_sites-1)
    hlist = twositeoperators(sites, localcfs, interactioncfs)
    #
    function links_odd(τ)
      return [exp(-im * τ * h) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(-im * τ * h) for h in hlist[2:2:end]]
    end

    # Simulazione
    # ===========
    # Determina lo stato iniziale a partire dalla stringa data nei parametri
    ψ₀ = parse_init_state(sites, parameters["chain_initial_state"])

    # Misuro le osservabili sullo stato iniziale
    current_allsites_ops = [[current(sites, s, j)
                             for j ∈ filter(n -> n != s, 1:n_sites)]
                            for s ∈ 1:n_sites]

    occn(ψ) = real.(expect(ψ, "N"))
    entropy(ψ) = real.([vonneumannentropy(ψ, sites, j) for j ∈ 2:n_sites-1])
    spincurrent(ψ) = reduce(vcat,
                        [real.([inner(ψ, j * ψ) for j in ops])
                         for ops ∈ current_allsites_ops])

    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout,
      normalisation,
      occnlist,
      entropylist,
      currentlist,
      ranks = evolve(ψ₀,
                     time_step_list,
                     parameters["skip_steps"],
                     parameters["TS_expansion_order"],
                     links_odd,
                     links_even,
                     parameters["MP_compression_error"],
                     parameters["MP_maximum_bond_dimension"];
                     fout=[norm, occn, entropy, spincurrent, linkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    entropylist = mapreduce(permutedims, vcat, entropylist)
    currentlist = mapreduce(permutedims, vcat, currentlist)
    ranks = mapreduce(permutedims, vcat, ranks)

    push!(timesteps_super, tout)
    push!(occ_n_super, occnlist)
    push!(bond_dimensions_super, ranks)
    push!(current_allsites_super, currentlist)
    push!(entropy_super, entropylist)
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
  N = size(occ_n_super[begin])[2]
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

  # Grafico dei ranghi del MPS
  # --------------------------
  N = size(bond_dimensions_super[begin])[2]
  plt = groupplot(timesteps_super,
                  bond_dimensions_super,
                  parameter_lists;
                  labels=hcat(["($j,$(j+1))" for j ∈ 1:N]...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"\chi_{k,k+1}(t)",
                  plottitle="Ranghi del MPS",
                  plotsize=plotsize)

  savefig(plt, "bond_dimensions.png")

  # Qui purtroppo bisogna assumere che tutte le simulazioni siano state
  # definite con lo stesso numero di spin... se no il seguente codice
  # si rompe.
  # `current_allsites_super` è una lista: ogni suo elemento si riferisce
  # a una simulazione differente.
  # Ogni suo elemento è una matrice di n_spins² colonne: la corrente dallo
  # spin i° allo spin k° si trova nella colonna (i-1)*n_spin + k.
  n_spins = parameter_lists[begin]["number_of_spin_sites"]
  for i ∈ 1:n_spins-1
    data = [table[:, (i-1)*n_spins .+ (1:n_spins)] for table ∈ current_allsites_super]
    plt = groupplot(timesteps_super,
                    data,
                    parameter_lists;
                    labels=reduce(hcat, ["l=$l" for l ∈ 1:n_spins]),
                    linestyles=reduce(hcat, repeat([:solid], n_spins)),
                    commonxlabel=L"\lambda\, t",
                    commonylabel=LaTeXString("\\langle j_{$i,l}\\rangle"),
                    plottitle="Corrente tra spin (dal $(i)° spin)",
                    plotsize=plotsize)

    savefig(plt, "spin_current_fromsite$i.png")
  end

  # Grafico dell'entropia di entanglement
  # -------------------------------------
  N = size(entropy_super[begin])[2]
  plt = groupplot(timesteps_super,
                  entropy_super,
                  parameter_lists;
                  labels=hcat(string.(1 .+ 1:N)...),
                  linestyles=hcat(repeat([:solid], N)...),
                  commonxlabel=L"\lambda\, t",
                  commonylabel=L"S_i(t)",
                  plottitle="Entropia di entanglement delle bipartizioni",
                  plotsize=plotsize)

  savefig(plt, "entropy.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
