#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

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
    current_allsites_ops = [[fermioncurrent(sites, s, j)
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

  # Plots
  # -----
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

  # Occupation numbers
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           occ_n_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.(1:N) ))
      push!(grp, ax)
    end
    pgfsave("occ_n.pdf", grp)
  end

  # Bond dimensions
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\chi_{i,i+1}(t)",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           bond_dimensions_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["($j,$(j+1))" for j ∈ 1:N] ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions.pdf", grp)
  end

  # Entanglement entropy
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"S_i(t)",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           entropy_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.(1 .+ (1:N)) ))
      push!(grp, ax)
    end
    pgfsave("entropy.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
