#!/usr/bin/julia

using PolyChaos, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using ITensors, PseudomodesTTEDOPA

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
  occn_TTEDOPA_super = []
  range_spins_super = []
  range_osc_left_super = []
  normalisation_TTEDOPA_super = []
  bond_dimensions_super = []
  oscmaxlevels_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    n_osc_left = parameters["number_of_oscillators_left"]
    n_osc_right = parameters["number_of_oscillators_right"]
    max_osc_dim = parameters["maximum_oscillator_space_dimension"]
    osc_dims_decay = parameters["oscillator_space_dimensions_decay"]
    sites = [
             reverse([siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_left, max_osc_dim, osc_dims_decay)]);
             repeat([siteind("S=1/2")], n_spin_sites);
             [siteind("Osc"; dim=d) for d ∈ oscdimensions(n_osc_right, max_osc_dim, osc_dims_decay)]
            ]
    for n ∈ eachindex(sites)
      sites[n] = addtags(sites[n], "n=$n")
    end

    range_osc_left = 1:n_osc_left
    range_spins = n_osc_left .+ (1:n_spin_sites)
    range_osc_right = n_osc_left .+ n_spin_sites .+ (1:n_osc_right)

    # Impostazione dei parametri
    # ==========================

    # - parametri tecnici
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    Ω = parameters["spectral_density_peak"]
    γ = parameters["spectral_density_half_width"]
    κ = parameters["spectral_density_overall_factor"]
    T = parameters["temperature"]
    ωc = parameters["frequency_cutoff"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    J(ω) = κ^2/π * 0.5γ * (1 / ((0.5γ)^2 + (ω-Ω)^2) - 1 / ((0.5γ)^2 + (ω+Ω)^2))
    Jtherm = ω -> thermalisedJ(J, ω, T)
    Jzero  = ω -> thermalisedJ(J, ω, 0)
    (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(Jtherm,
                                        (-ωc, 0, ωc),
                                        n_osc_left-1;
                                        Nquad=nquad,
                                        discretization=lanczos)
    (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(Jzero,
                                        (0, ωc),
                                        n_osc_right-1;
                                        Nquad=nquad,
                                        discretization=lanczos)

    # Raccolgo i coefficienti in due array (uno per quelli a sx, l'altro per
    # quelli a dx) per poterli disegnare assieme nei grafici.
    # (I coefficienti κ sono uno in meno degli Ω! Per ora pareggio le lunghezze
    # inserendo uno zero all'inizio dei κ…)
    osc_chain_coefficients_left = [Ωₗ [0; κₗ]]
    osc_chain_coefficients_right = [Ωᵣ [0; κᵣ]]

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

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto.
    ψ₀ = MPS(sites, [repeat(["0"], n_osc_left);
                     repeat(["Dn"], n_spin_sites);
                     repeat(["0"], n_osc_right)])

    # Osservabili da misurare
    # -----------------------
    occn(ψ) = real.(expect(ψ, "N")[1:range_spins[end]]) ./ norm(ψ)^2
    spinlinkdims(ψ) = [linkdims(ψ)[range_osc_left[end] : range_spins[end]];
                       maxlinkdim(ψ)]
    function oscmaxlevel(ψ)
      # Monitoro il livello massimo dell'operatore numero per i primi
      # quattro oscillatori della catena. Per farlo, prendo |⟨d-1ⁱ∣ψ⟩|², dove
      # ∣d-1ⁱ⟩ è lo stato più alto dell'oscillatore all'i° sito.
      nlev = []
      for ind ∈ last(range_osc_left, 4)
        ground = [repeat(["0"], n_osc_left);
                  repeat(["Dn"], n_spin_sites);
                  repeat(["0"], n_osc_right)]
        ground[ind] = string(ITensors.dim(sites[ind])-1)
        push!(nlev, ground)
      end
      maxn = [MPS(sites, l) for l ∈ nlev]
      return [abs2(inner(m, ψ)) for m ∈ maxn] ./ norm(ψ)^2
    end

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout, normalisation_TTEDOPA, occnlist, maxlevs, ranks = evolve(ψ₀,
                             time_step_list,
                             parameters["skip_steps"],
                             parameters["TS_expansion_order"],
                             links_odd,
                             links_even,
                             parameters["MP_compression_error"],
                             parameters["MP_maximum_bond_dimension"];
                             fout=[norm, occn, oscmaxlevel, spinlinkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occn_TTEDOPA = mapreduce(permutedims, vcat, occnlist)
    oscmaxlevels = mapreduce(permutedims, vcat, maxlevs)
    ranks = mapreduce(permutedims, vcat, ranks)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time_TTEDOPA => tout)
    sitelabels = [string.("L", reverse(range_osc_left));
                  string.("S", eachindex(range_spins));
                  string.("R", eachindex(range_osc_right))]
    for coln ∈ 1:size(occn_TTEDOPA, 2)
      sym = "occn_TTEDOPA_" * sitelabels[coln]
      push!(dict, Symbol(sym) => occn_TTEDOPA[:, coln])
    end
    for n ∈ -1:n_spin_sites-1
      from = sitelabels[range_spins[begin] + n]
      to   = sitelabels[range_spins[begin] + n+1]
      sym  = "rank_TTEDOPA_$from/$to"
      push!(dict, Symbol(sym) => ranks[:, n+2])
    end
    push!(dict, :maxrank_TTEDOPA => ranks[:, end])
    push!(dict, :norm_TTEDOPA => normalisation_TTEDOPA)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".TTEDOPA.dat")
    # Scrive la tabella su un file che ha la stessa estensione del file deITensors.i
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occn_TTEDOPA_super, occn_TTEDOPA)
    push!(normalisation_TTEDOPA_super, normalisation_TTEDOPA)
    push!(range_spins_super, range_spins)
    push!(range_osc_left_super, range_osc_left)
    push!(bond_dimensions_super, ranks)
    push!(oscmaxlevels_super, oscmaxlevels)
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

  # State norm
  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\Vert\psi(t)\Vert",
               title        = "State norm",
               "legend pos" = "outer north east",
"every axis plot/.append style" = "thick"
              })
    for (t, y, p, col) ∈ zip(timesteps_super,
                             normalisation_TTEDOPA_super,
                             parameter_lists,
                             distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("normalisation_TTEDOPA.pdf", ax)
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
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( [consecutivepairs(sitelabels); "max"] ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions_TTEDOPA.pdf", grp)
  end

  # Occupation numbers, left chain
  Nlines = 10
  occn_left = [occn[:, rn[end:-1:end-Nlines+1]]
               for (rn, occn) ∈ zip(range_osc_left_super, occn_TTEDOPA_super)]
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
      push!(ax, Legend( string.("L", 1:Nlines) ))
      push!(grp, ax)
    end
    pgfsave("population_left_chain_TTEDOPA.pdf", grp)
  end

  # Population of the highest level of the oscillators in the left chain
  Nlines = 4
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n^{(i)}_{{d_i}-1}(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           oscmaxlevels_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.("L", 1:Nlines) ))
      push!(grp, ax)
    end
    pgfsave("highest_level_population_left_chain_TTEDOPA.pdf", grp)
  end

  # Occupation numbers, spin chain
  occn_spin = [occn[:, rn]
               for (rn, occn) ∈ zip(range_spins_super, occn_TTEDOPA_super)]
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
    pgfsave("population_spin_chain_TTEDOPA.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
