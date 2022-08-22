#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
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
    @info "($current_sim_n di $tot_sim_n) Lettura dei parametri dal file."
    # Impostazione dei parametri
    # ==========================

    # - parametri per ITensors
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ₁ = parameters["oscillatorL1_spin_interaction_coefficient"]
    ω₁ = parameters["oscillatorL1_frequency"]
    γ₁ = parameters["oscillatorL1_damping_coefficient"]
    #
    κ₂ = parameters["oscillatorL2_spin_interaction_coefficient"]
    ω₂ = parameters["oscillatorL2_frequency"]
    γ₂ = parameters["oscillatorL2_damping_coefficient"]
    #
    κᵣ = parameters["oscillatorR_spin_interaction_coefficient"]
    ωᵣ = parameters["oscillatorR_frequency"]
    γᵣ = parameters["oscillatorR_damping_coefficient"]
    #
    η = parameters["oscillators_interaction_coefficient"]
    T = parameters["temperature"]
    oscdim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"] # deve essere un numero pari
    spin_range = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=oscdim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=oscdim)]

    # Definizione degli operatori nell'equazione di Lindblad
    # ===================================================
    # Calcolo dei coefficienti dell'Hamiltoniano trasformato
    @info "($current_sim_n di $tot_sim_n) Costruzione dell'operatore di evoluzione."
    n(T,ω) = T == 0 ? 0 : 1/(ℯ^(ω/T) - 1)

    ω̃₁ = (ω₁*κ₁^2 + ω₂*κ₂^2 + 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)
    ω̃₂ = (ω₂*κ₁^2 + ω₁*κ₂^2 - 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)

    κ̃₁ = sqrt(κ₁^2 + κ₂^2)
    κ̃₂ = ((ω₂-ω₁)*κ₁*κ₂ + η*(κ₁^2 - κ₂^2)) / (κ₁^2 + κ₂^2)

    γ̃₁⁺ = (κ₁^2*γ₁*(n(T,ω₁)+1) + κ₂^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    γ̃₁⁻ = (κ₁^2*γ₁* n(T,ω₁)    + κ₂^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)

    γ̃₂⁺ = (κ₂^2*γ₁*(n(T,ω₁)+1) + κ₁^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    γ̃₂⁻ = (κ₂^2*γ₁* n(T,ω₁)    + κ₁^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)

    γ̃₁₂⁺ = κ₁*κ₂*( γ₂*(n(T,ω₂)+1) - γ₁*(n(T,ω₁)+1) ) / (κ₁^2 + κ₂^2)
    γ̃₁₂⁻ = κ₁*κ₂*( γ₂*n(T,ω₂)     - γ₁*n(T,ω₁) )     / (κ₁^2 + κ₂^2)

    localcfs = zeros(Float64, length(sites))
    localcfs[1] = ω̃₂
    localcfs[2] = ω̃₁
    # L'evoluzione deve avvenire con l'operatore dell'ambiente isolato, quindi
    # "stacco" il sistema aperto mettendo a zero tutti i coefficienti che
    # non riguardano tale sottosistema. In questo modo non faccio fare calcoli
    # inutili al programma.
    interactioncfs = zeros(Float64, length(sites)-1)
    interactioncfs[1] = κ̃₂
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli operatori già creati gli operatori di dissipazione:
    # · per il primo oscillatore a sinistra,
    ℓlist[1] += γ̃₂⁺ * op("Lindb+", sites[1]) * op("Id", sites[2])
    ℓlist[1] += γ̃₂⁻ * op("Lindb-", sites[1]) * op("Id", sites[2])
    # · per il secondo oscillatore (occhio che non essendo più all'estremo
    #   della catena questo operatore viene diviso tra ℓ₁,₂ e ℓ₂,₃),
    ℓlist[1] += 0.5γ̃₁⁺ * op("Id", sites[1]) * op("Lindb+", sites[2])
    ℓlist[1] += 0.5γ̃₁⁻ * op("Id", sites[1]) * op("Lindb-", sites[2])
    ℓlist[2] += 0.5γ̃₁⁺ * op("Lindb+", sites[2]) * op("Id", sites[3])
    ℓlist[2] += 0.5γ̃₁⁻ * op("Lindb-", sites[2]) * op("Id", sites[3])
    # · l'operatore misto su (1) e (2),
    ℓlist[1] += (γ̃₁₂⁺ * mixedlindbladplus(sites[1], sites[2]) +
                 γ̃₁₂⁻ * mixedlindbladminus(sites[1], sites[2]))
    # · infine per l'oscillatore a destra, come al solito,
    ℓlist[end] += γᵣ * (op("Id", sites[end-1]) *
                        op("Damping", sites[end]; ω=ωᵣ, T=0))
    #
    function links_odd(τ)
      return [exp(τ * ℓ) for ℓ in ℓlist[1:2:end]]
    end
    function links_even(τ)
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
    # 1) per gli oscillatori
    if T != 0 # ρ₀ = Z⁻¹exp(-βH)
      HoscL = (ω̃₁ * num(oscdim) ⊗ id(oscdim) +
               ω̃₂ * id(oscdim) ⊗ num(oscdim) +
               κ̃₂ * (a⁺(oscdim) ⊗ a⁻(oscdim) +
                     a⁻(oscdim) ⊗ a⁺(oscdim)))
      M = exp(-1/T * HoscL)
      M /= tr(M)
    else # ρ₀ = |∅⟩ ⟨∅| = |0⟩ ⟨0| ⊗ |0⟩ ⟨0|
      emptystate = zeros(Float64, oscdim, oscdim)
      emptystate[1, 1] = 1.0
      M = emptystate ⊗ emptystate
    end
    P = (id(oscdim) ⊗ (a⁺(oscdim) + a⁻(oscdim))) * M
    # 2) la vettorizzo sul prodotto delle basi hermitiane dei due siti
    v = vec(P, [êᵢ ⊗ êⱼ for (êᵢ, êⱼ) ∈ [Base.product(gellmannbasis(oscdim),
                                                     gellmannbasis(oscdim))...]])
    # 3) inserisco il vettore in un tensore con gli Index degli oscillatori
    iv = itensor(v, sites[1], sites[2])
    # 4) lo decompongo in due pezzi con una SVD
    f1, f2, _, _ = factorize(iv, sites[1]; which_decomp="svd")
    # 5) rinomino il Link tra i due fattori come "Link,l=1" anziché
    #    "Link,fact" che è il Tag assegnato da `factorize`
    replacetags!(f1, "fact" => "l=1")
    replacetags!(f2, "fact" => "l=1")

    X₀ρ₀ = chain(MPS([f1, f2]),
               parse_init_state(sites[spin_range],
                                parameters["chain_initial_state"]),
               #MPS([state(sites[end-1], "0")]),
               MPS([state(sites[end],   "0")]))

    # Osservabili
    # -----------
    X₀ = MPS(sites, [i == 2 ? "vecX" : "vecId" for i ∈ eachindex(sites)])
    correlation(ρ) = κ̃₁^2 * real(inner(X₀, ρ))

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    tout, correlationlist = evolve(X₀ρ₀,
                                   time_step_list,
                                   parameters["skip_steps"],
                                   parameters["TS_expansion_order"],
                                   links_odd,
                                   links_even,
                                   parameters["MP_compression_error"],
                                   parameters["MP_maximum_bond_dimension"];
                                   fout=[correlation])
    
    @info "($current_sim_n di $tot_sim_n) Scrittura dei file di output."
    #expXcorrelation = [c₀ * ℯ^(-γ*t) * cos(ω₁*t) for t ∈ tout]
    #Xcorrelationlist = hcat(correlationlist, expXcorrelation)

    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    #push!(dict, :correlation_calc => Xcorrelationlist[:,1])
    push!(dict, :correlation_exp => correlationlist)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(Xcorrelation_super, correlationlist)
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

  # Correlation function
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle X(t)X(0)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           Xcorrelation_super,
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
      push!(ax, Legend( ["calculated", "expected"] ))
      push!(grp, ax)
    end
    pgfsave("Xcorrelation.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
