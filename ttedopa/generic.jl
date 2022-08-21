#!/usr/bin/julia

using ITensors, DataFrames, LaTeXStrings, CSV, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

let
  parameter_lists = load_parameters(ARGS)
  tot_sim_n = length(parameter_lists)

  # Se il primo argomento da riga di comando è una cartella (che 
  # dovrebbe contenere i file dei parametri), mi sposto subito in tale
  # posizione in modo che i file di output, come grafici e tabelle,
  # siano salvati insieme ai file di parametri.
  prev_dir = pwd()
  if isdir(ARGS[1])
    cd(ARGS[1])
  end

  # Le seguenti liste conterranno i risultati della simulazione
  # per ciascuna lista di parametri fornita.
  timesteps_super = []
  occn_super = []
  spin_current_super = []
  bond_dimensions_super = []
  spin_chain_levels_super = []
  range_osc_left_super = []
  range_spins_super = []
  range_osc_right_super = []
  osc_chain_coefficients_left_super = []
  osc_chain_coefficients_right_super = []
  snapshot_super = []
  normalisation_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================
    # Numero di siti e dimensioni
    n_spin_sites = parameters["number_of_spin_sites"]

    n_osc_left = parameters["left_bath"]["number_of_oscillators"]
    max_osc_dim_left = parameters["left_bath"]["maximum_oscillator_space_dimension"]
    osc_dims_decay_left = parameters["left_bath"]["oscillator_space_dimensions_decay"]

    n_osc_right = parameters["right_bath"]["number_of_oscillators"]
    max_osc_dim_right = parameters["right_bath"]["maximum_oscillator_space_dimension"]
    osc_dims_decay_right = parameters["right_bath"]["oscillator_space_dimensions_decay"]

    sites = [
             reverse([siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_left, max_osc_dim_left, osc_dims_decay_left)]);
             repeat([siteind("S=1/2")], n_spin_sites);
             [siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_right, max_osc_dim_right, osc_dims_decay_right)]
            ]
    for n ∈ eachindex(sites)
      sites[n] = addtags(sites[n], "n=$n")
    end

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    # Parametri tecnici per la simulazione
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # Istanti di tempo da attraversare
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Parametri della catena di spin
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    
    (Ωₗ, κₗ, ηₗ) = getchaincoefficients(parameters["left_bath"])
    (Ωᵣ, κᵣ, ηᵣ) = getchaincoefficients(parameters["right_bath"])

    # Raccolgo i coefficienti in due array (uno per quelli a sx, l'altro per
    # quelli a dx) per poterli disegnare assieme nei grafici.
    # (I coefficienti κ sono uno in meno degli Ω! Pareggio le lunghezze
    # inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ωₗ [0; κₗ]]
    osc_chain_coefficients_right = [Ωᵣ [0; κᵣ]]

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#

    localcfs = [reverse(Ωₗ); repeat([ε], n_spin_sites); Ωᵣ]
    interactioncfs = [reverse(κₗ); ηₗ; repeat([1], n_spin_sites-1); ηᵣ; κᵣ]
    hlist = twositeoperators(sites, localcfs, interactioncfs)
    #
    function links_odd(τ)
      return [exp(-im * τ * h) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(-im * τ * h) for h in hlist[2:2:end]]
    end

    # Osservabili da misurare
    # =======================
    # - la corrente di spin
    # (Ci metto anche quella tra spin e primi oscillatori)
    spin_current_ops = [-2ηₗ*current(sites,
                                     range_spins[1]-1,
                                     range_spins[1]);
                        [current(sites, j, j+1)
                         for j ∈ range_spins[1:end-1]];
                        -2ηᵣ*current(sites,
                                     range_spins[end],
                                     range_spins[end]+1)]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto; mi riservo di decidere
    # volta per volta come inizializzare l'oscillatore più a destra nella
    # catena a sinistra, per motivi di diagnostica.
    osc_sx_init_state = MPS(sites[range_osc_left], "0")
    spin_init_state = parse_init_state(sites[range_spins],
                                       parameters["chain_initial_state"])
    osc_dx_init_state = MPS(sites[range_osc_right], "0")
    ψ₀ = chain(osc_sx_init_state, spin_init_state, osc_dx_init_state)

    # Osservabili da misurare
    # -----------------------
    occn(ψ) = real.(expect(ψ, "N")) ./ norm(ψ)^2
    spincurrent(ψ) = real.([inner(ψ', j, ψ) for j ∈ spin_current_ops]) ./ norm(ψ)^2
    # Calcolo i ranghi tra tutti gli spin, più quelli tra gli spin e
    # i primi oscillatori, quelli appena attaccati alla catena.
    # Questo risolve anche il problema di come trattare questa funzione
    # quando c'è un solo spin nella catena.
    spinlinkdims(ψ) = [linkdims(ψ)[range_osc_left[end] : range_spins[end]];
                       maxlinkdim(ψ)]

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout, normalisation, occnlist, spincurrentlist, ranks = evolve(ψ₀,
                             time_step_list,
                             parameters["skip_steps"],
                             parameters["TS_expansion_order"],
                             links_odd,
                             links_even,
                             parameters["MP_compression_error"],
                             parameters["MP_maximum_bond_dimension"];
                             fout=[norm, occn, spincurrent, spinlinkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    spincurrentlist = mapreduce(permutedims, vcat, spincurrentlist)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    sitelabels = [string.("L", n_osc_left:-1:1);
                  string.("S", 1:n_spin_sites);
                  string.("R", 1:n_osc_right)]
    for (n, label) ∈ enumerate(sitelabels)
      push!(dict, Symbol(string("occn_", label)) => occnlist[:, n])
    end
    for n ∈ -1:n_spin_sites-1
      from = sitelabels[range_spins[begin] + n]
      to   = sitelabels[range_spins[begin] + n+1]
      sym  = "current_adjsites_$from/$to"
      push!(dict, Symbol(sym) => spincurrentlist[:, n+2])
      sym  = "rank_$from/$to"
      push!(dict, Symbol(sym) => ranks[:, n+2])
    end
    push!(dict, :maxrank => ranks[:, end])
    push!(dict, :norm => normalisation)

    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occn_super, occnlist)
    push!(spin_current_super, spincurrentlist)
    push!(bond_dimensions_super, ranks)
    push!(range_osc_left_super, range_osc_left)
    push!(range_spins_super, range_spins)
    push!(range_osc_right_super, range_osc_right)
    push!(osc_chain_coefficients_left_super, osc_chain_coefficients_left)
    push!(osc_chain_coefficients_right_super, osc_chain_coefficients_right)
    push!(snapshot_super, occnlist[end,:])
    push!(normalisation_super, normalisation)
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

  # Occupation numbers, left chain
  occn_left = [mat[:, range]
               for (mat, range) ∈ zip(occn_super, range_osc_left_super)]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           occn_left,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.("L", N:-1:1) ))
      push!(grp, ax)
    end
    pgfsave("population_left_chain.pdf", grp)
  end

  # Occupation numbers, spin chain
  occn_spin = [mat[:, range]
               for (mat, range) ∈ zip(occn_super, range_spins_super)]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           occn_spin,
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
    pgfsave("population_spin_chain.pdf", grp)
  end
  
  # Occupation numbers, right chain
  occn_right = [mat[:, range]
                for (mat, range) ∈ zip(occn_super, range_osc_right_super)]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           occn_right,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.("R", 1:N) ))
      push!(grp, ax)
    end
    pgfsave("population_right_chain.pdf", grp)
  end

  # Occupation numbers, totals
  sums = [[sum(mat[:, rangeL], dims=2);;
           sum(mat[:, rangeS], dims=2);;
           sum(mat[:, rangeR], dims=2);;
           sum(mat, dims=2)]
          for (mat, rangeL, rangeS, rangeR) ∈ zip(occn_super,
                                                  range_osc_left_super,
                                                  range_spins_super,
                                                  range_osc_right_super)]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           sums,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(4),
                           [repeat(["solid"], 3); "dashed"])
        plot = Plot({ color = c, $ls }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["left ch.", "spin ch.", "right ch.", "total"] ))
      push!(grp, ax)
    end
    pgfsave("population_totals.pdf", grp)
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
      nsites = size(data, 2) - 2
      sitelabels = ["L1"; string.("S", 1:nsites); "R1"]
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(nsites+2),
                           [repeat(["solid"], nsites+1); "dashed"])
        plot = Plot({ color = c, $ls }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( [consecutivepairs(sitelabels); "max"] ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions.pdf", grp)
  end

  # Particle current, between adjacent spin sites
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle j_{i,i+1}(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           spin_current_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      nspins = size(data, 2) - 1
      sitelabels = ["L"; string.("S", 1:nspins); "R"]
      for (y, c) ∈ zip(eachcol(data), readablecolours(nsites+1))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(sitelabels) ))
      push!(grp, ax)
    end
    pgfsave("particle_current_adjacent_sites.pdf", grp)
  end

  # Chain map coefficients, left chain
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"i",
        ylabel = "Coefficient",
    })
    for (i, data, p) ∈ zip([1:p["number_of_oscillators_left"]
                            for p ∈ parameter_lists],
                           osc_chain_coefficients_left_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      for (y, c) ∈ zip(eachcol(data), readablecolours(2))
        plot = Plot({ color = c }, Table([i, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( [L"\Omega_i", L"\kappa_i"] ))
      push!(grp, ax)
    end
    pgfsave("left_chain_coefficients.pdf", grp)
  end

  # Chain map coefficients, right chain
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"i",
        ylabel = "Coefficient",
    })
    for (i, data, p) ∈ zip([1:p["number_of_oscillators_right"]
                            for p ∈ parameter_lists],
                           osc_chain_coefficients_right_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      for (y, c) ∈ zip(eachcol(data), readablecolours(2))
        plot = Plot({ color = c }, Table([i, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( [L"\Omega_i", L"\kappa_i"] ))
      push!(grp, ax)
    end
    pgfsave("right_chain_coefficients.pdf", grp)
  end

  # Snapshot of occupation numbers at the end
  @pgf begin
    ax = Axis({
               xlabel       = L"i",
               ylabel       = L"n_i(t_\mathrm{end})",
               title        = "Snapshot of the population at the end",
               "legend pos" = "outer north east"
              })
    for (i, y, p, col) ∈ zip(vcat.(range_osc_left_super,
                                   range_spins_super,
                                   range_osc_right_super),
                             snapshot_super,
                             parameter_lists,
                             distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([i, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("snapshot.pdf", ax)
  end

  # Snapshot of occupation numbers at the end, spins only
  @pgf begin
    ax = Axis({
               xlabel       = L"i",
               ylabel       = L"n_i(t_\mathrm{end})",
               title        = "Snapshot of the spins population at the end",
               "legend pos" = "outer north east"
              })
    for (i, y, p, col) ∈ zip(range_spins_super,
                             getindex.(snapshot_super, range_spins_super),
                             parameter_lists,
                             distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([i, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("snapshot_spins.pdf", ax)
  end

  # State norm
  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\Vert\psi(t)\Vert",
               title        = "State norm",
               "legend pos" = "outer north east"
              })
    for (t, y, p, col) ∈ zip(timesteps_super,
                             normalisation_super,
                             parameter_lists,
                             distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("normalisation.pdf", ax)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
