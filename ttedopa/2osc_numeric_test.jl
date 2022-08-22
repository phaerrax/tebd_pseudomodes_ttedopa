#!/usr/bin/julia

using QuantumOptics, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

let
  @info "Caricamento dei parametri delle simulazioni."
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
  numeric_occn_super = []
  numeric_tsteps_super = []
  norm_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
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
    oscdim = parameters["maximum_oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    n_osc_left = 2
    n_osc_right = 1

    # Definizione degli operatori nell'Hamiltoniana
    # =============================================
    # Calcolo dei coefficienti dalla densità spettrale
    @info "Simulazione $current_sim_n di $tot_sim_n: calcolo dei coefficienti TEDOPA."
    J(ω) = κ^2 * 0.5γ/π * (hypot(0.5γ, ω-Ω)^(-2) - hypot(0.5γ, ω+Ω)^(-2))
    Jtherm = ω -> thermalisedJ(J, ω, T)
    Jzero  = ω -> thermalisedJ(J, ω, 0)
    (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(Jtherm,
                                        (-ωc, 0, ωc),
                                        2;
                                        Nquad=nquad,
                                        discretization=lanczos)
    (Ωᵣ, κᵣ, ηᵣ) = chainmapcoefficients(Jzero,
                                        (0, ωc),
                                        2;
                                        Nquad=nquad,
                                        discretization=lanczos)

    @info "Simulazione $current_sim_n di $tot_sim_n: calcolo della soluzione dell'equazione di Schrödinger."
    bosc = FockBasis(oscdim)
    bspin = SpinBasis(1//2)
    bcoll = tensor(bosc, bosc,
                   [bspin for _ ∈ 1:n_spin_sites]...,
                   bosc, bosc)

    # Costruisco i vari operatori per l'Hamiltoniano
    function num(i::Int)
      if i ∈ 1:2
        op = embed(bcoll, i, number(bosc))
      elseif i ∈ 2 .+ (1:n_spin_sites)
        op = embed(bcoll, i, 0.5*(sigmaz(bspin) + one(bspin)))
      elseif i ∈ 2 .+ n_spin_sites .+ (1:2)
        op = embed(bcoll, i, number(bosc))
      else
        throw(DomainError(i, "$i not in range"))
      end
      return op
    end

    hspinloc(i) = embed(bcoll, 2+i, sigmaz(bspin))
    hspinint(i) = tensor(one(bosc),
                         one(bosc),
                         [one(bspin) for i ∈ 1:i-1]...,
                         sigmap(bspin)⊗sigmam(bspin)+sigmam(bspin)⊗sigmap(bspin),
                         [one(bspin) for i ∈ 1:n_spin_sites-i-1]...,
                         one(bosc),
                         one(bosc))

    # Gli operatori Hamiltoniani
    # · oscillatori a sinistra
    Hoscsx = (Ωₗ[2] * num(1) +
              Ωₗ[1] * num(2) +
              κₗ[1] * tensor(create(bosc)⊗destroy(bosc)+destroy(bosc)⊗create(bosc),
                             [one(bspin) for _ ∈ 1:n_spin_sites]...,
                             one(bosc),
                             one(bosc)))

    # · interazione tra oscillatore a sinistra e spin
    Hintsx = ηₗ * tensor(one(bosc),
                         create(bosc)+destroy(bosc),
                         sigmax(bspin),
                         [one(bspin) for _ ∈ 1:n_spin_sites-1]...,
                         one(bosc),
                         one(bosc))

    # · catena di spin
    Hspin = 0.5ε*sum(hspinloc.(1:n_spin_sites)) -
            0.5*sum(hspinint.(1:n_spin_sites-1))

    # · interazione tra oscillatore a destra e spin
    Hintdx = ηᵣ * tensor(one(bosc),
                         one(bosc),
                         [one(bspin) for _ ∈ 1:n_spin_sites-1]...,
                         sigmax(bspin),
                         create(bosc)+destroy(bosc),
                         one(bosc))
    # · oscillatori a destra
    Hoscdx = (Ωᵣ[1] * num(2+n_spin_sites+1) +
              Ωᵣ[2] * num(2+n_spin_sites+2) +
              κᵣ[1] * tensor(one(bosc),
                             one(bosc),
                             [one(bspin) for _ ∈ 1:n_spin_sites]...,
                             create(bosc)⊗destroy(bosc)+destroy(bosc)⊗create(bosc)))

    H = Hoscsx + Hintsx + Hspin + Hintdx + Hoscdx

    # Stato iniziale (vuoto)
    ψ₀ = tensor(fockstate(bosc, 0),
                fockstate(bosc, 0),
                [spindown(bspin) for _ ∈ 1:n_spin_sites]...,
                fockstate(bosc, 0),
                fockstate(bosc, 0))

    function fout(t, ψ)
      occn = [real(QuantumOptics.expect(N, ψ))
               for N ∈ num.(1:(2+n_spin_sites+2))]
      return [occn; QuantumOptics.norm(ψ)]
    end
    tout, output = timeevolution.schroedinger(time_step_list, ψ₀, H; fout=fout)
    output = mapreduce(permutedims, vcat, output)
    numeric_occn = output[:, 1:end-1]
    statenorm = output[:, end]

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(numeric_occn_super, numeric_occn)
    push!(norm_super, statenorm)

    dict = Dict(:time => tout)
    sitelabels = ["L2"; "L1"; string.("S", 1:n_spin_sites); "R1"; "R2"]
    for (j, label) ∈ enumerate(sitelabels)
      push!(dict, Symbol(string("occn_", label)) => numeric_occn[:,j])
    end
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "_numeric.dat")
    CSV.write(filename, table)
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

  # Occupation numbers
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           numeric_occn_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N),
                           ["dashed";
                            "dashed";
                            repeat(["solid"], N-4);
                            "dashed";
                            "dashed"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( ["L2" "L1"; string.(1:N-4); "R1"; "R2"] ))
      push!(grp, ax)
    end
    pgfsave("population_numeric_solution.pdf", grp)
  end

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
                             normalisation_super,
                             parameter_lists,
                             distinguishable_colors(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("normalisation_numeric_solution.pdf", ax)
  end

  # Population of spin sites
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           [occn[:, 3:end-2] for occn ∈ numeric_occn_super],
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
    pgfsave("population_spin_sites.pdf", grp)
  end

  # Population, left pseudomodes
  data = [[occn[:, 1];;
           occn[:, 2];;
           occn[:, 1] .+ occn[: ,2]]
           for occn ∈ numeric_occn_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           data,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N)
                           ["solid", "solid", "dashed"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( ["L2", "L1", "L1+L2"] ))
      push!(grp, ax)
    end
    pgfsave("population_left_pmode_numeric_solution.pdf", grp)
  end
  
  # Population, right pseudomodes
  data = [[occn[:, end-1];;
           occn[:, end];;
           occn[:, end-1] .+ occn[:, end]]
           for occn ∈ numeric_occn_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           data,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N)
                           ["solid", "solid", "dashed"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( ["R1", "R2", "R1+R2"] ))
      push!(grp, ax)
    end
    pgfsave("population_right_pmode_numeric_solution.pdf", grp)
  end
 
  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
