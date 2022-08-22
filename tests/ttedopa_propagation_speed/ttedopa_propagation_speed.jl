#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma mostra come evolve un'eccitazione solitaria in una catena
# di oscillatori ottenuta con il T-TEDOPA, in modo da determinare una lunghezza
# ottimale per la catena nel resto delle simulazioni.
# Confronterò l'evoluzione di un sistema con i coefficienti T-TEDOPA con quella
# di un sistema in cui ωₖ e κₖ sono sostituiti dai loro valori asintotici.

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

  timeranges = []
  chainranges = []
  frequencies = []
  couplingcoeffs = []
  first2sitesevos = []
  firstsiteevos = []
  asymptoticevos = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================
    # - parametri tecnici
    nquad = Int(parameters["PolyChaos_nquad"])
    ωc = parameters["frequency_cutoff"]
    nosc = parameters["number_of_oscillators_left"]

    # - parametri fisici
    Ω₀ = parameters["spectral_density_peak"]
    T = parameters["temperature"]
    γ = parameters["spectral_density_half_width"]
    κ = parameters["spectral_density_overall_factor"]

    # Calcolo dei coefficienti dalla densità spettrale
    Θ(x) = x ≥ 0 ? 1 : 0
    thermf(T,ω) = T == 0 ? Θ(ω) : 0.5*(coth(0.5ω/T) + 1)
    J(T,ω) = thermf(T,ω) * κ^2 * 0.5γ/π * (hypot(0.5γ, ω-Ω₀)^(-2) - hypot(0.5γ, ω+Ω₀)^(-2))
    if T != 0
      support = (-ωc, 0, ωc)
    else
      support = (0, ωc)
    end
    (Ω, κ, η) = chainmapcoefficients(ω -> J(T,ω),
                                     support,
                                     nosc-1;
                                     Nquad=nquad,
                                     discretization=lanczos)

    push!(chainranges, 1:nosc)
    push!(frequencies, Ω)
    push!(couplingcoeffs, κ)

    H(ωs, κs) = diagm(0 => ωs) + diagm(-1 => κs) + diagm(1 => κs)
    U(τ,H) = exp(-im * τ * H)

    s₁ = zeros(ComplexF64, nosc)
    s₁[1] = one(eltype(s₁))
    s₂ = zeros(ComplexF64, nosc)
    s₂[2] = one(eltype(s₂))
    
    occn1(ψ) = abs2(dot(s₁, ψ))
    occn2(ψ) = abs2(dot(s₂, ψ))

    t = 0:(parameters["simulation_time_step"]):(parameters["simulation_end_time"])
    ureal = U(step(t), H(Ω,κ))
    uasym = U(step(t), H(repeat([Ω[end]], nosc), repeat([κ[end]], nosc-1)))
    nreal1 = similar(t)
    nreal2 = similar(t)
    nasym1 = similar(t)
    nasym2 = similar(t)
    ψasym = s₁
    ψreal = s₁
    for i ∈ eachindex(t)
      nreal1[i], nreal2[i] = occn1(ψreal), occn2(ψreal)
      nasym1[i], nasym2[i] = occn1(ψasym), occn2(ψasym)
      ψreal = ureal*ψreal
      ψasym = uasym*ψasym
    end
    push!(timeranges, t)
    push!(firstsiteevos, nreal1)
    push!(first2sitesevos, nreal1 .+ nreal2)
    push!(asymptoticevos, nasym1)

    dict = Dict(:time => collect(t))
    push!(dict, :occn_real1 => nreal1)
    push!(dict, :occn_real2 => nreal2)
    push!(dict, :occn_asym1 => nasym1)
    push!(dict, :occn_asym2 => nasym2)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    CSV.write(filename, table)

    dict = Dict(:freq => Ω)
    push!(dict, :coupling => [η; κ])
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "_coeffs.dat")
    CSV.write(filename, table)
  end

  # Plots
  # -----
  @info "Drawing plots."

  @pgf begin
    ax = Axis({
               xlabel       = L"i",
               ylabel       = L"\Omega_i",
               "legend pos" = "outer north east",
"every axis plot/.append style" = "thick"
               title        = "T-TEDOPA frequency coefficients",
              })
    for (t, y, p, col) ∈ zip(chainranges,
                             frequencies,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("frequencies.pdf", ax)
  end

  @pgf begin
    ax = Axis({
               xlabel       = L"i",
               ylabel       = L"\kappa_i",
               "legend pos" = "outer north east",
"every axis plot/.append style" = "thick"
               title        = "T-TEDOPA interaction coefficients",
              })
    for (t, y, p, col) ∈ zip([rn[1:end-1] for rn ∈ chainranges],
                             couplingcoeffs,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("couplingcoeffs.pdf", ax)
  end

  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\langle n_1(t)\rangle",
               "legend pos" = "outer north east",
"every axis plot/.append style" = "thick"
               title        = "Evolution using real coefficients",
              })
    for (t, y, p, col) ∈ zip(timeranges,
                             firstsitesevos,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("realevos1.pdf", ax)
  end

  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\langle n_1(t)+n_2(t)\rangle",
               "legend pos" = "outer north east",
"every axis plot/.append style" = "thick"
               title        = "Evolution using real coefficients",
              })
    for (t, y, p, col) ∈ zip(timeranges,
                             first2sitesevos,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("realevos1+2.pdf", ax)
  end

  @pgf begin
    ax = Axis({
               xlabel       = L"\lambda t",
               ylabel       = L"\langle n_1(t)\rangle",
               "legend pos" = "outer north east",
"every axis plot/.append style" = "thick"
               title        = "Evolution using asymptotic coefficients",
              })
    for (t, y, p, col) ∈ zip(timeranges,
                             asymptoticevos,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("asymptoticevos.pdf", ax)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
end
