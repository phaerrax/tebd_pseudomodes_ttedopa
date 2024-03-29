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
  coherence_super = []
  bond_dimensions_super = []
  osc_chain_coefficients_left_super = []
  snapshot_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri tecnici
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    λ = parameters["spin_interaction_energy"]
    Ω = parameters["ohmic_decay_factor"]
    a = parameters["ohmic_exponent"]
    b = parameters["ohmic_mult_factor"]
    ωc = parameters["frequency_cutoff"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    n_osc_left = parameters["number_of_oscillators_left"]
    sites = [siteinds("Osc", n_osc_left; dim=osc_dim);
             siteinds("S=1/2", n_spin_sites)]

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    @info "Calcolo dei parametri TEDOPA in corso."
    support = (0, ωc)
    J(ω) = b * ℯ^(-ω/Ω) * (ω/Ω)^a

    (Ω, κ, η) = chainmapcoefficients(J, support, n_osc_left-1; Nquad=nquad)
    @assert length(Ω) == n_osc_left
    @assert length(κ) == n_osc_left-1
    # Mi assicuro di avere il numero giusto di coefficienti (magari per
    # sbaglio ne faccio generare a PolyChaos uno in più o in meno: in
    # questo modo mi assicuro che la lunghezza sia quella corretta.)
    
    # Raccolgo i coefficienti in un array per poterli disegnare assieme nei
    # grafici. (I coefficienti κ sono uno in meno degli Ω! Per ora
    # pareggio le lunghezze inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ω [0; κ]]

    localcfs = [reverse(Ω); repeat([ε], n_spin_sites)]
    interactioncfs = [reverse(κ); η; repeat([1], n_spin_sites-1)]
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

    # Osservabili da misurare
    # =======================
    # - la corrente di spin
    list = spin_current_op_list(sites[range_spins])
    spin_current_ops = [embed_slice(sites, range_spins, j)
                        for j in list]
    # - l'occupazione degli autospazi dell'operatore numero
    projectors = [level_subspace_proj(sites[range_spins], n)
                  for n = 0:n_spin_sites]
    num_eigenspace_projs = [embed_slice(sites, range_spins, p)
                            for p in projectors]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto
    osc_sx_init_state = MPS(sites[range_osc_left], "0")
    spin_init_state = MPS(sites[range_spins], "X+")
    current_state = chain(osc_sx_init_state, spin_init_state)

    # Misura della coerenza
    function spincoherence(ψ::MPS)
      # Lo spin si trova sull'ultimo sito del MPS
      orthogonalize!(ψ, length(ψ))
      ρ = matrix(last(ψ) * dag(prime(last(ψ), "Site")))
      return abs(tr(ρ * [0 1; 0 0]))
    end

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [expect(current_state, "N")]
    bond_dimensions = [linkdims(current_state)]
    spin_current = [[real(inner(current_state, j * current_state))
                     for j in spin_current_ops]]
    spin_chain_levels = [levels(num_eigenspace_projs,
                                current_state)]
    coherence = Real[spincoherence(current_state)]

    # Evoluzione temporale
    # --------------------
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1

    @time begin
      for _ in time_step_list[2:end]
        current_state = apply(evo,
                              current_state;
                              cutoff=max_err,
                              maxdim=max_dim)
        if skip_count % skip_steps == 0
          push!(occ_n, expect(current_state, "N"))
          push!(coherence, spincoherence(current_state))
          push!(bond_dimensions, linkdims(current_state))
        end
        next!(progress)
        skip_count += 1
      end
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
    push!(coherence_super, coherence)
    push!(bond_dimensions_super, permutedims(hcat(bond_dimensions...)))
    push!(osc_chain_coefficients_left_super, osc_chain_coefficients_left)
    push!(snapshot_super, snapshot)
  end

  # Plots
  # -----
  @info "Drawing plots."

  # Common options for group plots
  nrows = Int(ceil(tot_sim_n / 2))
  common_opts = @pgf {
    no_markers,
    legend_pos = "outer north east",
    "every axis plot/.append style" = "thick"
   }
  group_opts = @pgf {
    group_style = {
      group_size        = "$nrows by 2",
      y_descriptions_at = "edge left",
      horizontal_sep    = "2cm",
      vertical_sep      = "2cm"
    },
    common_opts...
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

  # Coherence of the spin's reduced density matrix
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"(\rho_S)_{+,-}(t)",
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
      push!(grp, ax)
    end
    pgfsave("coherences.pdf", grp)
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
      sitelabels = [string.(N-1:-1:1); "S"]
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(sitelabels) ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions.pdf", grp)
  end

  # Chain map coefficients
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"i",
        ylabel = "Coefficient",
       })
    for (i, data, p) ∈ zip([reverse(1:length(chain[:,1]))
                            for chain ∈ osc_chain_coefficients_left_super],
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
    pgfsave("chain_coefficients.pdf", grp)
  end

  # Snapshot of occupation numbers at the end
  @pgf begin
    ax = Axis({
               xlabel       = L"i",
               ylabel       = L"n_i(t_\mathrm{end})",
               title        = "Snapshot of the population at the end",
               common_opts...
              })
    inds = [reverse(range_osc_left); range_spins]
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
