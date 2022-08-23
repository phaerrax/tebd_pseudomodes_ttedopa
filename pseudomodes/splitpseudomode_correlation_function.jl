#!/usr/bin/julia

using LinearAlgebra, LaTeXStrings, DataFrames, CSV, QuadGK, PGFPlotsX, Colors
using ITensors, PseudomodesTTEDOPA

disablegrifqtech()

let  
const ⊗ = kron

  @info "Lettura dei file con i parametri."
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
    @info "($current_sim_n di $tot_sim_n) Lettura dei parametri dal file."

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
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    Ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]

    n(T,ω) = T == 0 ? 0.0 : 1/expm1(ω/T)
    J(ω) = κ^2 * 0.5γₗ/π * (hypot(0.5γₗ, ω-Ω)^(-2) - hypot(0.5γₗ, ω+Ω)^(-2))

    # ITensors internal parameters
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    oscdim = parameters["oscillator_space_dimension"]
    oscdim_single = oscdim*3

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # +-----------------------------------------------------+ 
    # | Part 1: correlation function of a single pseudomode |
    # +-----------------------------------------------------+
    n_spin_sites = parameters["number_of_spin_sites"]
    range_spins = 1 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 1; dim=oscdim_single);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=oscdim_single)]

    localcfs = zeros(Float64, length(sites))
    localcfs[1] = Ω
    interactioncfs = zeros(Float64, length(sites)-1)
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli estremi della catena gli operatori di dissipazione
    ℓlist[begin] += γₗ * (op("Damping", sites[begin]; ω=Ω, T=T) *
                          op("Id", sites[begin+1]))
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
                        op("Damping", sites[end]; ω=Ω, T=0))
    #
    function links_odd1(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even1(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Lo stato iniziale qui è X₀ρ₀.
    # In ρ₀, l'oscillatore sx è in equilibrio termico, quello dx è vuoto.
    # Lo stato iniziale della catena è dato da "chain_initial_state".
    X₀ρ₀ = chain(MPS([state(sites[1], "X⋅Therm"; ω=Ω, T=T)]),
                 parse_init_state(sites[range_spins],
                                  parameters["chain_initial_state"]),
                 MPS([state(sites[end], "0")]))

    # Osservabili
    # -----------
    X₀ = MPS(sites, [i == 1 ? "vecX" : "vecId" for i ∈ eachindex(sites)])
    correlation_single(ρ) = κ^2 * inner(X₀, ρ)

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout, singleXcorrelation = evolve(X₀ρ₀,
                                    time_step_list,
                                    parameters["skip_steps"],
                                    parameters["TS_expansion_order"],
                                    links_odd1,
                                    links_even1,
                                    parameters["MP_compression_error"],
                                    parameters["MP_maximum_bond_dimension"];
                                    fout=[correlation_single])
    dict = Dict(:time => tout)

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

    push!(dict, :correlation_true_re  => real.(trueXcorrelation))
    push!(dict, :correlation_true_im  => imag.(trueXcorrelation))

    push!(dict, :correlation_pmodeexp_re  => real.(pmodeexpXcorrelation))
    push!(dict, :correlation_pmodeexp_im  => imag.(pmodeexpXcorrelation))

    push!(dict, :correlation_single_re => real.(singleXcorrelation))
    push!(dict, :correlation_single_im => imag.(singleXcorrelation))

    # +----------------------------------------------------+ 
    # | Part 2: correlation function of a split pseudomode |
    # +----------------------------------------------------+
    # Set new initial oscillator states as empty.
    parameters["left_oscillator_initial_state"] = "empty"

    range_spins = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=oscdim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=oscdim)]

    # Transform the pseudomode into two zero-temperature new modes.
    κ₁ = κ * sqrt( 1+n(T,Ω) )
    ω₁ = Ω
    γ₁ = γₗ
    #
    κ₂ = κ * sqrt( n(T,Ω) )
    ω₂ = -Ω
    γ₂ = γₗ
    #
    κᵣ = κ
    ωᵣ = Ω
    #γᵣ = γᵣ
    #
    η = 0

    # Definizione degli operatori nell'equazione di Lindblad
    # ======================================================
    # Calcolo i coefficienti dell'Hamiltoniano trasformato.
    ω̃₁ = (ω₁*κ₁^2 + ω₂*κ₂^2 + 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)
    ω̃₂ = (ω₂*κ₁^2 + ω₁*κ₂^2 - 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)
    κ̃₁ = sqrt(κ₁^2 + κ₂^2)
    κ̃₂ = ((ω₂-ω₁)*κ₁*κ₂ + η*(κ₁^2 - κ₂^2)) / (κ₁^2 + κ₂^2)
    # Dato che T = 0 dappertutto, posso semplificare qualcosa.
    #γ̃₁⁺ = (κ₁^2*γ₁*(n(T,ω₁)+1) + κ₂^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    #γ̃₁⁻ = (κ₁^2*γ₁* n(T,ω₁)    + κ₂^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)
    #γ̃₂⁺ = (κ₂^2*γ₁*(n(T,ω₁)+1) + κ₁^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    #γ̃₂⁻ = (κ₂^2*γ₁* n(T,ω₁)    + κ₁^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)
    #γ̃₁₂⁺ = κ₁*κ₂*( γ₂*(n(T,ω₂)+1) - γ₁*(n(T,ω₁)+1) )  / (κ₁^2 + κ₂^2)
    #γ̃₁₂⁻ = κ₁*κ₂*( γ₂*n(T,ω₂)     - γ₁*n(T,ω₁) )      / (κ₁^2 + κ₂^2)
    γ̃₁⁺ = γ₁ # == γ₂
    γ̃₁⁻ = 0
    γ̃₂⁺ = γ₂
    γ̃₂⁻ = 0
    γ̃₁₂⁺ = 0 # perché γ₁ = γ₂
    γ̃₁₂⁻ = 0

    localcfs = zeros(Float64, length(sites))
    localcfs[1] = ω̃₂
    localcfs[2] = ω̃₁
    interactioncfs = zeros(Float64, length(sites)-1)
    interactioncfs[1] = κ̃₂

    @info "($current_sim_n di $tot_sim_n) Costruzione degli operatori di evoluzione temporale."
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli operatori già creati gli operatori di dissipazione:
    # rimuovo direttamente quelli nulli.
    # · per il primo oscillatore a sinistra,
    ℓlist[1] += γ̃₂⁺ * op("Lindb+", sites[1]) * op("Id", sites[2])
    #ℓlist[1] += γ̃₂⁻ * op("Lindb-", sites[1]) * op("Id", sites[2])
    # · per il secondo oscillatore (occhio che non essendo più all'estremo
    #   della catena questo operatore viene diviso tra ℓ₁,₂ e ℓ₂,₃),
    ℓlist[1] += 0.5γ̃₁⁺ * op("Id", sites[1]) * op("Lindb+", sites[2])
    #ℓlist[1] += 0.5γ̃₁⁻ * op("Id", sites[1]) * op("Lindb-", sites[2])
    ℓlist[2] += 0.5γ̃₁⁺ * op("Lindb+", sites[2]) * op("Id", sites[3])
    #ℓlist[2] += 0.5γ̃₁⁻ * op("Lindb-", sites[2]) * op("Id", sites[3])
    # · l'operatore misto su (1) e (2),
    #ℓlist[1] += γ̃₁₂⁺ * mixedlindbladplus(sites[1], sites[2])
    #ℓlist[1] += γ̃₁₂⁻ * mixedlindbladminus(sites[1], sites[2])
    # · infine per l'oscillatore a destra, come al solito,
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
                        op("Damping", sites[end]; ω=ωᵣ, T=0))
    #
    function links_odd2(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even2(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[2:2:end]]
    end

    # La correlazione
    # ===============
    # Detto X l'operatore dell'oscillatore "L1", la funzione di correlazione
    # da calcolare è
    # ⟨XₜX₀⟩ = tr(ρ₀XₜX₀) = vec(Xₜ)'vec(X₀ρ₀);
    # l'evoluzione di X avviene secondo l'equazione di Lindblad aggiunta,
    # che vettorizzata ha L' al posto di L:
    # vec(Xₜ) = exp(tL')vec(X₀).
    # Allora
    # ⟨XₜX₀⟩ = vec(X₀)'exp(tL')'vec(X₀ρ₀) = vec(X₀)'exp(tL)vec(X₀ρ₀).

    # Simulazione
    # ===========
    @info "($current_sim_n di $tot_sim_n) Costruzione dello stato iniziale."
    # Stato iniziale
    # --------------
    # Lo stato iniziale qui è X₀ρ₀ (vedi eq. sopra).
    # Per calcolare lo stato iniziale dei due oscillatori a sinistra:
    # 1) la temperatura è sempre 0 quindi lo stato iniziale è
    #    ρ₀ = |∅⟩ ⟨∅| = |0⟩ ⟨0| ⊗ |0⟩ ⟨0|
    a⁻(d::Int)  = diagm(1  => sqrt.(1:dim-1))
    a⁺(d::Int)  = diagm(-1 => sqrt.(1:dim-1))
    id(d::Int)  = Matrix{Int}(I, dim, dim)
    emptystate = zeros(Float64, oscdim, oscdim)
    emptystate[1, 1] = 1.0
    M = (id(oscdim) ⊗ (a⁺(oscdim) + a⁻(oscdim))) * (emptystate ⊗ emptystate)
    # 2) la vettorizzo sul prodotto delle basi hermitiane dei due siti
    v = PseudomodesTTEDOPA.vec(M, [êᵢ ⊗ êⱼ for (êᵢ, êⱼ) ∈ [Base.product(gellmannbasis(oscdim), gellmannbasis(oscdim))...]])
    # 3) inserisco il vettore in un tensore con gli Index degli oscillatori
    iv = itensor(v, sites[1], sites[2])
    # 4) lo decompongo in due pezzi con una SVD
    f1, f2, _, _ = factorize(iv, sites[1]; which_decomp="svd")
    # 5) rinomino il Link tra i due fattori come "Link,l=1" anziché
    #    "Link,fact" che è il Tag assegnato da `factorize`
    replacetags!(f1, "fact" => "l=1")
    replacetags!(f2, "fact" => "l=1")

    X₀ρ₀ = chain(MPS([f1, f2]),
               parse_init_state(sites[range_spins],
                                parameters["chain_initial_state"]),
               MPS([state(sites[end], "0")]))

    # Osservabili
    # -----------
    X₀ = MPS(sites, [i == 2 ? "vecX" : "vecId" for i ∈ eachindex(sites)])
    correlation_split(ρ) = κ̃₁^2 * inner(X₀, ρ)

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout, splitXcorrelation = evolve(X₀ρ₀,
                                    time_step_list,
                                    parameters["skip_steps"],
                                    parameters["TS_expansion_order"],
                                    links_odd2,
                                    links_even2,
                                    parameters["MP_compression_error"],
                                    parameters["MP_maximum_bond_dimension"];
                                    fout=[correlation_split])

    @info "($current_sim_n di $tot_sim_n) Scrittura dei file di output."

    push!(dict, :correlation_split_re => real.(splitXcorrelation))
    push!(dict, :correlation_split_im => imag.(splitXcorrelation))

    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(Xcorrelation_super, hcat(singleXcorrelation,
                                   splitXcorrelation,
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
    legend_pos = "outer north east",
"every axis plot/.append style" = "thick"
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
                           ["solid", "solid", "dashed", "dotted"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( ["single p.mode",
                         "split p.mode",
                         "p.mode expected",
                         "true"] ))
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
                           ["solid", "solid", "dashed", "dotted"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( ["single p.mode",
                         "split p.mode",
                         "p.mode expected",
                         "true"] ))
      push!(grp, ax)
    end
    pgfsave("Xcorrelation_im.pdf", grp)
  end

  # Correlation function, comparison
  datas = [[real.(Xcorrelation[:,1]) .- real.(Xcorrelation[:,2]);;
            imag.(Xcorrelation[:,1]) .- imag.(Xcorrelation[:,2])]
           for Xcorrelation ∈ Xcorrelation_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle X(t)X(0)\rangle_\mathrm{single}-\langle X(t)X(0)\rangle_\mathrm{split}",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           datas,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N),
                           ["solid", "dashed"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( ["real part", "imag part"] ))
      push!(grp, ax)
    end
    pgfsave("Xcorrelation_diff.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
