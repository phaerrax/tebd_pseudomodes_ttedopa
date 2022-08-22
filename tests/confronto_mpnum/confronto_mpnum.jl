#!/usr/bin/julia

using ITensors, DataFrames, LaTeXStrings, CSV, PGFPlotsX, Colors
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
  occ_n_comp_super = []
  snapshot_comp_super = []
  ranks_super = []
  timesteps_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    n_sites = parameters["number_of_spin_sites"] 
    ε = parameters["spin_excitation_energy"]
    # λ = 1

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)

    # Leggo il file con i risultati di Python
    pythoncsv = CSV.File(replace(parameters["filename"],
                                 ".json" => "_python.csv"))
    occ_n_py = hcat([pythoncsv["occ_n$N"] for N∈1:n_sites]...)

    # Costruzione della catena
    # ========================
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
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             time_step,
                             parameters["TS_expansion_order"])

    # Simulazione
    # ===========
    # Determina lo stato iniziale a partire dalla stringa data nei parametri
    current_state = parse_init_state(sites,
                                     parameters["chain_initial_state"])

    # Misuro le osservabili sullo stato iniziale
    nums = expect(current_state, "N")
    occ_n = [nums]
    occ_n_comp = [nums - occ_n_py[1,:]]
    ranks = [linkdims(current_state)]

    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    for (j, _) in zip(2:length(time_step_list), time_step_list[2:end])
      current_state = apply(evo,
                            current_state;
                            cutoff=max_err,
                            maxdim=max_dim)
      nums = expect(current_state, "N")
      push!(occ_n, nums)
      push!(occ_n_comp, nums - occ_n_py[j,:])
      push!(ranks, linkdims(current_state))
      next!(progress)
    end

    snapshot_jl = occ_n[end]
    snapshot_py = [last(pythoncsv["occ_n$N"])
                   for N ∈ 1:n_sites]

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list)
    push!(occ_n_super, hcat(occ_n...)')
    push!(occ_n_comp_super, hcat(occ_n_comp...)')
    push!(snapshot_comp_super, [snapshot_jl snapshot_py])
    push!(ranks_super, hcat(ranks...)')
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
    legend_pos = "outer north east",
"every axis plot/.append style" = "thick"
  }

  # Occupation numbers, left chain
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
    pgfsave("population.pdf", grp)
  end

  # Comparison btw. mpnum and ITensors on selected sites
  cols = [1, 5, 10]
  selection = [mat[:, cols] for mat ∈ occ_n_comp_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\Delta\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           selection,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2) # whatever this is
      for (y, c, ls) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( cols ))
      push!(grp, ax)
    end
    pgfsave("itensors_mpnum_discrepancy.pdf", grp)
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
      N = size(data, 2) # whatever this is
      for (y, c, ls) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(1:N) ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions.pdf", grp)
  end

  # Snapshot of the occupation numbers at the end, ITensors v. mpnum comparison
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = "Site",
        ylabel = "Occ. n. comparison",
    })
    for (i, data, p) ∈ zip([1:size(arr, 1) for arr ∈ snapshot_comp_super],
                           snapshot_comp_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      for (y, c) ∈ zip(eachcol(data), readablecolours(2))
        plot = Plot({ color = c }, Table([i, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["ITensors" "mpnum"] ))
      push!(grp, ax)
    end
    pgfsave("snapshot_comparison.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
