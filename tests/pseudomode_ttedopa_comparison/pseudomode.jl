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

  timesteps_super = []
  occn_PM_super = []
  normalisation_PM_super = []
  range_spins_super = []
  bond_dimensions_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["oscillator_spin_interaction_coefficient"]
    if (haskey(parameters, "oscillator_damping_coefficient_left") &&
        haskey(parameters, "oscillator_damping_coefficient_right"))
      γₗ = parameters["oscillator_damping_coefficient_left"]
      γᵣ = parameters["oscillator_damping_coefficient_right"]
    elseif haskey(parameters, "oscillator_damping_coefficient")
      γₗ = parameters["oscillator_damping_coefficient"]
      γᵣ = γₗ
    else
      throw(ErrorException("Oscillator damping coefficient not provided."))
    end
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    range_spins = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=osc_dim)]

    #= Definizione degli operatori nell'equazione di Lindblad
    ======================================================
    I siti del sistema sono numerati come segue:
    | 1 | 2 | ... | n_spin_sites | n_spin_sites+1 | n_spin_sites+2 |
    ↑   │                        │          ↑
    │   └───────────┬────────────┘          │
    │               │                       │
    │        catena di spin                 │
    oscillatore sx                    oscillatore dx
    =#
    localcfs = [ω; repeat([ε], n_spin_sites); ω]
    interactioncfs = [κ; repeat([1], n_spin_sites-1); κ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += γₗ * (op("Damping", sites[begin]; ω=ω, T=T) *
                         op("Id", sites[begin+1]))
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
                       op("Damping", sites[end]; ω=ω, T=0))
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # Osservabili da misurare
    # =======================
    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ eachindex(sites)])
                   for n ∈ eachindex(sites)]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # L'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    ρ₀ = chain(parse_init_state_osc(sites[1], "thermal"; ω=ω, T=T),
               parse_init_state(sites[range_spins], "empty"),
               parse_init_state_osc(sites[end], "empty"))

    # Osservabili
    # -----------
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)
    spinlinkdims(ρ) = [linkdims(ρ); maxlinkdim(ρ)]

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout,
      normalisation,
      occnlist,
      ranks = evolve(ρ₀,
                     time_step_list,
                     parameters["skip_steps"],
                     parameters["TS_expansion_order"],
                     links_odd,
                     links_even,
                     parameters["MP_compression_error"],
                     parameters["MP_maximum_bond_dimension"];
                     fout=[trace, occn, spinlinkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occn_PM = mapreduce(permutedims, vcat, occnlist)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    sitelabels = ["L"; string.("S", eachindex(range_spins)); "R"]
    for (j, label) ∈ enumerate(sitelabels)
      push!(dict, Symbol("occn_PM_" * label) => occn_PM[:, j])
    end
    for n ∈ eachindex(sitelabels)[1:end-1]
      from = sitelabels[n]
      to   = sitelabels[n+1]
      sym  = "rank_PM_$from/$to"
      push!(dict, Symbol(sym) => ranks[:, n])
    end
    push!(dict, :norm_PM => normalisation)
    push!(dict, :maxrank_PM => ranks[:, end])
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".PM.dat")
    CSV.write(filename, table)

    push!(timesteps_super, tout)
    push!(occn_PM_super, occn_PM)
    push!(normalisation_PM_super, normalisation)
    push!(range_spins_super, range_spins)
    push!(bond_dimensions_super, ranks)
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
      N = size(data, 2) # = no. of spins + 2
      sitelabels=["L"; string.("S", 1:N-2); "R"]
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N),
                           [repeat(["solid"], N-1); "dashed"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( [consecutivepairs(sitelabels); "max"] ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions_pseudomodes.pdf", grp)
  end


  # Population of spin sites
  spinsonly = [occn[:, rn] for (rn, occn) ∈ zip(range_spins_super,
                                                occn_PM_super)]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           spinsonly,
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
    pgfsave("population_spin_sites_pseudomodes.pdf", grp)
  end

  # Trace of the density matrix
  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\mathrm{tr}\rho(t)",
               "legend pos" = "outer north east",
"every axis plot/.append style" = "thick"
              })
    for (t, y, p, col) ∈ zip(timesteps_super,
                             normalisation_PM,super,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("normalisation_pseudomodes.pdf", ax)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
