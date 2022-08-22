#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, QuadGK, PGFPlotsX, Colors
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
  Xcorrelation_super = []

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
    # La densità spettrale è data da
    # J(ω) = κ² ⋅ γ/2π ⋅ 1/(γ²/4 + (ω-Ω)²)
    T = parameters["temperature"]
    ωc = parameters["frequency_cutoff"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

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

    #= Definizione degli operatori nell'Hamiltoniana
       =============================================
       I siti del sistema sono numerati come segue:
       - 1:n_osc_left -> catena di oscillatori a sinistra
       - n_osc_left+1:n_osc_left+n_spin_sites -> catena di spin
       - n_osc_left+n_spin_sites+1:end -> catena di oscillatori a destra
    =#
    # Calcolo dei coefficienti dalla densità spettrale
    J(ω) = κ^2 * 0.5γ/π * (1 / ((0.5γ)^2 + (ω-Ω)^2) - 1 / ((0.5γ)^2 + (ω+Ω)^2))
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

    localcfs = [reverse(Ωₗ); repeat([0.0], n_spin_sites + n_osc_right)]
    interactioncfs = [reverse(κₗ); 0.0;
                      repeat([0.0], n_spin_sites-1);
                      0.0; repeat([0.0], length(κᵣ))]
    hlist = twositeoperators(sites, localcfs, interactioncfs)
    #
    function links_odd(τ)
      return [exp(-im * τ * h) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(-im * τ * h) for h in hlist[2:2:end]]
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
    # Lo stato iniziale qui è X₀ρ₀ (vedi eq. sopra), e ρ₀ è lo stato di
    # vuoto del sistema: tutto nello stato fondamentale.
    # 
    # c(t) = ⟨ψ₀∣ Xₜ X₀ ∣ψ₀⟩ = ⟨ψ₀∣ X₀ Φₜ ( X₀ ∣ψ₀⟩ ).
    osc_sx_init_state = MPS(sites[range_osc_left],  "0")
    spin_init_state   = MPS(sites[range_spins],     "Dn")
    osc_dx_init_state = MPS(sites[range_osc_right], "0")
    ψ₀ = chain(osc_sx_init_state, spin_init_state, osc_dx_init_state)

    X₀ψ₀ = apply(MPO(sites,
                     [i == range_osc_left[end] ? "X" : "Id"
                      for i ∈ eachindex(sites)]),
                 ψ₀)

    # Osservabili
    # -----------
    X₀ = embed_slice(sites,
                     range_osc_left[end:end],
                     MPO(sites[range_osc_left[end:end]], "X"))
    correlation(ψ)::ComplexF64 = ηₗ^2 * inner(ψ₀', X₀, ψ)
    # Devo esplicitare che il risultato è un numero complesso, altrimenti il
    # programma si lamenta che si vuole inserire un numero complesso in un
    # vettore di Float64 (visto che il primissimo elemento risulta reale).

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout, calcXcorrelation = evolve(X₀ψ₀,
                                    time_step_list,
                                    parameters["skip_steps"],
                                    parameters["TS_expansion_order"],
                                    links_odd,
                                    links_even,
                                    parameters["MP_compression_error"],
                                    parameters["MP_maximum_bond_dimension"];
                                    fout=[correlation])

    n(T,Ω) = T == 0 ? 0 : expm1(Ω/T)^(-1)
    function c(t)
      quadgk(ω -> J(ω) * (ℯ^(-im*ω*t) * (1+n(T,ω)) + ℯ^(im*ω*t) * n(T,ω)), 0, Inf)[1]
    end
    function cᴿ(κ, Ω, γ, T, t)
      if T != 0
        κ^2 * (coth(0.5*Ω/T)*cos(Ω*t) - im*sin(Ω*t)) * ℯ^(-0.5γ*t)
      else
        κ^2 * ℯ^(-im*Ω*t - 0.5γ*t)
      end
    end
    expXcorrelation = c.(tout)
    Xcorrelation = hcat(calcXcorrelation, expXcorrelation)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    push!(dict, :correlation_calc_re => real.(calcXcorrelation))
    push!(dict, :correlation_calc_im => imag.(calcXcorrelation))
    push!(dict, :correlation_exp_re  => real.(expXcorrelation))
    push!(dict, :correlation_exp_im  => imag.(expXcorrelation))
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(Xcorrelation_super, Xcorrelation)
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

  # Correlation function, real part
  data = [[real.(Xcorrelation[:,1])  real.(Xcorrelation[:,2])]
          for Xcorrelation ∈ Xcorrelation_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\mathrm{Re}\,(\langle X_tX_0\rangle)",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           data,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["calculated", "expected"] ))
      push!(grp, ax)
    end
    pgfsave("Xcorrelation_re.pdf", grp)
  end

  # Correlation function, imaginary part
  data = [[imag.(Xcorrelation[:,1])  imag.(Xcorrelation[:,2])]
          for Xcorrelation ∈ Xcorrelation_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\mathrm{Im}\,(\langle X_tX_0\rangle)",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           data,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["calculated", "expected"] ))
      push!(grp, ax)
    end
    pgfsave("Xcorrelation_im.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
