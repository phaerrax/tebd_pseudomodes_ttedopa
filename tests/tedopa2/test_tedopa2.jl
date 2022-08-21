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
  occ_n_super = []
  bond_dimensions_super = []
  osc_chain_coefficients_super = []
  snapshot_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri tecnici
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε1 = parameters["spin_excitation_energy"]
    ε2 = 0.0
    K = parameters["spin_interaction_energy"]
    g = parameters["frequency_cutoff"]
    osc_dim = parameters["oscillator_space_dimension"]
    λ = parameters["spectral_density_prop_constant"]
    γ = parameters["spectral_density_half_width"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    n_osc_left = parameters["number_of_oscillators_left"]
    n_osc_right = parameters["number_of_oscillators_right"]
    sites = [siteinds("Osc", n_osc_left; dim=osc_dim);
             siteinds("S=1/2", n_spin_sites);
             siteinds("Osc", n_osc_right; dim=osc_dim)]

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    @info "Calcolo dei parametri TEDOPA in corso."
    support = (0, g)
    J(x) = 8λ * γ * x / (x^2 + γ^2)

    (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(J,
                                        support,
                                        g,
                                        n_osc_left-1;
                                        Nquad=nquad,
                                        discretization=lanczos)
    (Ωᵣ, κᵣ, ηᵣ) = (Ωₗ, κₗ, ηₗ)
    # In caso di bisogno, le righe seguenti prendono i coefficienti
    # di ricorsione calcolati da py-tedopa per la funzione spettrale
    # J(x) = 8λ * γ * x / (x^2 + γ^2)
    #@assert n_osc_left == n_osc_left
    #α = parse.(Float64, readlines("alphas_lorentz.dat"))
    #β = parse.(Float64, readlines("betas_lorentz.dat"))
    #Ω = g .* α[1:n_osc_right]
    #κ = g .* sqrt.(β[2:n_osc_right])
    #η = sqrt(β[1])
    #(Ωₗ, κₗ, ηₗ) = (Ω, κ, η)
    #(Ωᵣ, κᵣ, ηᵣ) = (Ω, κ, η)

    # Raccolgo i coefficienti in array per poterli disegnare assieme nei
    # grafici. (I coefficienti κ sono uno in meno degli Ω! Per ora
    # pareggio le lunghezze inserendoci η all'inizio...)
    osc_chain_coefficients = [Ωₗ [ηₗ; κₗ] Ωᵣ [ηᵣ; κᵣ]]

    localcfs = [reverse(Ωₗ); [ε1, ε2]; Ωᵣ]
    interactioncfs = [reverse(κₗ); [ηₗ, K, ηᵣ]; κᵣ]
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
                             τ,
                             parameters["TS_expansion_order"])

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto
    current_state = chain(MPS(sites[range_osc_left], "0"),
                          MPS(sites[range_spins], ["Up", "Dn"]),
                          MPS(sites[range_osc_right], "0"))

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [expect(current_state, "N")]
    bond_dimensions = [linkdims(current_state)]

    # Evoluzione temporale
    # --------------------
    @info "Avvio della simulazione."
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1

    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state;
                            cutoff=max_err,
                            maxdim=max_dim)
      if skip_count % skip_steps == 0
        push!(occ_n, expect(current_state, "N"))
        push!(bond_dimensions, linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    snapshot = occ_n[end]

    ## Creo una tabella con i dati rilevanti da scrivere nel file di output
    #dict = Dict(:time => time_step_list[1:skip_steps:end])
    #tmp_list = hcat(occ_n...)
    #for (j, name) in enumerate([[Symbol("occ_n_left$n") for n∈n_osc_left:-1:1];
    #                            [Symbol("occ_n_spin$n") for n∈1:n_spin_sites];
    #                            [Symbol("occ_n_right$n") for n∈1:n_osc_right]])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(spin_current...)
    #for (j, name) in enumerate([Symbol("spin_current$n")
    #                            for n = 1:n_spin_sites-1])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(spin_chain_levels...)
    #for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(bond_dimensions...)
    #len = n_osc_left + n_spin_sites + n_osc_right
    #for (j, name) in enumerate([Symbol("bond_dim$n") for n ∈ 1:len-1])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #table = DataFrame(dict)
    #filename = replace(parameters["filename"], ".json" => "") * ".dat"
    ## Scrive la tabella su un file che ha la stessa estensione del file dei
    ## parametri, con estensione modificata.
    #CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list[1:skip_steps:end])
    push!(occ_n_super, permutedims(hcat(occ_n...)))
    push!(bond_dimensions_super, permutedims(hcat(bond_dimensions...)))
    push!(osc_chain_coefficients_super, osc_chain_coefficients)
    push!(snapshot_super, snapshot)
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
      push!(ax, Legend( [string.(N-1:-1:1); "S"] ))
      push!(grp, ax)
    end
    pgfsave("populations.pdf", grp)
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
      push!(ax, Legend( consecutivepairs(1:N) ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions.pdf", grp)
  end

  # Chain map coefficients, both chains
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"i",
        ylabel = "Coefficient",
       })
    for (i, data, p) ∈ zip([reverse(1:length(chain[:,1]))
                            for chain ∈ osc_chain_coefficients_super],
                           [mat[:, 1:2]
                            for mat ∈ osc_chain_coefficients_super],
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      for (y, c) ∈ zip(eachcol(data), readablecolours(2))
        plot = Plot({ color = c }, Table([i, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( [L"\Omega_i", L"\kappa_i"] ))
      push!(grp, ax)
    end
    pgfsave("chain_coefficients.pdf", grp)
  end

  # Snapshot of occupation numbers at the end
  @pgf begin
    ax = Axis({
               xlabel       = L"i",
               ylabel       = L"n_i(t_\mathrm{end})",
               title        = "Snapshot of the population at the end",
               "legend pos" = "outer north east"
              })
    inds = [reverse(range_osc_left); range_spins; range_osc_right]
    for (y, p, col) ∈ zip(snapshot_super,
                          parameter_lists,
                          distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([inds, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("snapshot.pdf", ax)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
