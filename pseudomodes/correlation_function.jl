#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, QuadGK, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

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
  Xcorrelation_super = []

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
    Ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]
    J(ω) = κ^2 * 0.5γₗ/π * (hypot(0.5γₗ, ω-Ω)^(-2) - hypot(0.5γₗ, ω+Ω)^(-2))

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
    spin_range = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=osc_dim)]

    # Definizione degli operatori nell'equazione di Lindblad
    # ===================================================
    localcfs = zeros(Float64, n_spin_sites+2)
    localcfs[1] = Ω
    interactioncfs = zeros(Float64, n_spin_sites+1)
    # Devo porre κ=0 in modo che l'evoluzione sia quella dello
    # pseudomodo isolato.
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += γₗ * (op("Damping", sites[begin]; ω=Ω, T=T) *
                          op("Id", sites[begin+1]))
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
                        op("Damping", sites[end]; ω=Ω, T=0))
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # La correlazione
    # ===============
    # ⟨XₜX₀⟩ = tr(ρ₀XₜX₀) = vec(Xₜ)'vec(X₀ρ₀);
    # l'evoluzione di X avviene secondo l'equazione di Lindblad aggiunta,
    # che vettorizzata ha L' al posto di L:
    # vec(Xₜ) = exp(tL')vec(X₀).
    # Allora
    # ⟨XₜX₀⟩ = vec(X₀)'exp(tL')'vec(X₀ρ₀) = vec(X₀)'exp(tL)vec(X₀ρ₀).

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Lo stato iniziale qui è X₀ρ₀ (vedi eq. sopra).
    # In ρ₀, l'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    X₀ρ₀ = chain(MPS([state(sites[1], "X⋅Therm"; ω=Ω, T=T)]),
                 parse_init_state(sites[spin_range],
                                  parameters["chain_initial_state"]),
                 MPS([state(sites[end], "0")]))

    # Osservabili
    # -----------
    X₀ = MPS(sites, [i == 1 ? "vecX" : "vecId" for i ∈ eachindex(sites)])
    correlation(ρ) = κ^2 * inner(X₀, ρ)

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout, calcXcorrelation = evolve(X₀ρ₀,
                                    time_step_list,
                                    parameters["skip_steps"],
                                    parameters["TS_expansion_order"],
                                    links_odd,
                                    links_even,
                                    parameters["MP_compression_error"],
                                    parameters["MP_maximum_bond_dimension"];
                                    fout=[correlation])

    n(T,ω) = T == 0 ? 0.0 : 1/expm1(ω/T)
    function c(t) # Funzione di correlazione esatta
      quadgk(ω -> J(ω) * (ℯ^(-im*ω*t) * (1+n(T,ω)) + ℯ^(im*ω*t) * n(T,ω)), 0, Inf)[1]
    end
    function cᴿ(κ, Ω, γ, T, t) # Funzione di correlazione attesa per gli p.modi
      if T != 0
        κ^2 * (coth(0.5*Ω/T)*cos(Ω*t) - im*sin(Ω*t)) * ℯ^(-0.5γ*t)
      else
        κ^2 * ℯ^(-im*Ω*t - 0.5γ*t)
      end
    end
    pmodeexpXcorrelation = [cᴿ(κ,Ω,γₗ,T, t) for t ∈ tout]
    trueXcorrelation = c.(tout)

    dict = Dict(:time => tout)

    push!(dict, :correlation_true_re  => real.(trueXcorrelation))
    push!(dict, :correlation_true_im  => imag.(trueXcorrelation))

    push!(dict, :correlation_pmodeexp_re  => real.(pmodeexpXcorrelation))
    push!(dict, :correlation_pmodeexp_im  => imag.(pmodeexpXcorrelation))

    push!(dict, :correlation_calc_re => real.(calcXcorrelation))
    push!(dict, :correlation_calc_im => imag.(calcXcorrelation))

    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(Xcorrelation_super, hcat(calcXcorrelation,
                                   pmodeexpXcorrelation,
                                   trueXcorrelation))
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

  # Correlation function, real part
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\mathrm{Re}\,(\langle X_tX_0\rangle)",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           real.(Xcorrelation_super),
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N),
                           ["solid", "dashed", "dotted"])
        plot = Plot({ color = c, $ls }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["calculated", "expected (p.mode)", "true"] ))
      push!(grp, ax)
    end
    pgfsave("Xcorrelation_re.pdf", grp)
  end

  # Correlation function, imaginary part
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\mathrm{Im}\,(\langle X_tX_0\rangle)",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           imag.(Xcorrelation_super),
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N),
                           ["solid", "dashed", "dotted"])
        plot = Plot({ color = c, $ls }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["calculated", "expected (p.mode)", "true"] ))
      push!(grp, ax)
    end
    pgfsave("Xcorrelation_im.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
