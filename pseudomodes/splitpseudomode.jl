#!/usr/bin/julia

using LinearAlgebra, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using ITensors, PseudomodesTTEDOPA

const ⊗ = kron

disablegrifqtech()

let  
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
  occn_super = []
  originaloccn_super = []
  originaloccn_yrange_super = []
  current_adjsites_super = []
  bond_dimensions_super = []
  osclevelsL1_super = []
  osclevelsL2_super = []
  normalisation_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    @info "($current_sim_n di $tot_sim_n) Costruzione degli operatori di evoluzione temporale."

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

    n(T,ω) = T == 0 ? 0.0 : (ℯ^(ω/T)-1)^(-1)

    # Set new initial oscillator states as empty.
    parameters["left_oscillator_initial_state"] = "empty"

    # ITensors internal parameters
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]

    # Transform the pseudomode into two zero-temperature new modes.
    ε = parameters["spin_excitation_energy"]
    # λ = 1
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
    T = 0
    if (haskey(parameters, "hot_oscillator_space_dimension") &&
        haskey(parameters, "cold_oscillator_space_dimension"))
      hotoscdim = parameters["hot_oscillator_space_dimension"]
      coldoscdim = parameters["cold_oscillator_space_dimension"]
    else
      hotoscdim = parameters["oscillator_space_dimension"]
      coldoscdim = parameters["oscillator_space_dimension"]
    end

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    range_spins = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=hotoscdim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=coldoscdim)]

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

    localcfs = [ω̃₂; ω̃₁; repeat([ε], n_spin_sites); ωᵣ]
    interactioncfs = [κ̃₂; κ̃₁; repeat([1], n_spin_sites-1); κᵣ]
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
                       [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                   for n ∈ 1:length(sites)]

    # - la normalizzazione (cioè la traccia) della matrice densità
    full_trace = MPS(sites, "vecId")

    current_adjsites_ops = [fermioncurrent(sites, j, j+1)
                            for j ∈ range_spins[1:end-1]]

    osclevels_projs_L1 = [embed_slice(sites, 2:2, osc_levels_proj(sites[2], n))
                           for n ∈ 0:hotoscdim-1]
    osclevels_projs_L2 = [embed_slice(sites, 1:1, osc_levels_proj(sites[1], n))
                           for n ∈ 0:hotoscdim-1]

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    @info "($current_sim_n di $tot_sim_n) Creazione dello stato iniziale."
    if parameters["left_oscillator_initial_state"] == "thermal"
      # L'ambiente sx è in equilibrio termico, quello dx è vuoto.
      # Lo stato iniziale della catena è dato da "chain_initial_state".
      # Per calcolare lo stato iniziale dei due oscillatori a sinistra:
      # 1) Calcolo la matrice densità dello stato termico
      a⁻(d::Int)  = diagm(1  => sqrt.(1:dim-1))
      a⁺(d::Int)  = diagm(-1 => sqrt.(1:dim-1))
      num(d::Int) = diagm(0  => 0:dim-1)
      id(d::Int)  = Matrix{Int}(I, dim, dim)
      HoscL = (ω̃₁ * num(hotoscdim) ⊗ id(hotoscdim) +
               ω̃₂ * id(hotoscdim) ⊗ num(hotoscdim) +
               κ̃₂ * (a⁺(hotoscdim) ⊗ a⁻(hotoscdim) +
                     a⁻(hotoscdim) ⊗ a⁺(hotoscdim)))
      M = exp(-1/T * HoscL)
      M /= tr(M)
      # 2) la vettorizzo sul prodotto delle basi hermitiane dei due siti
      v = PseudomodesTTEDOPA.vec(M, [êᵢ ⊗ êⱼ for (êᵢ, êⱼ) ∈ [Base.product(gellmannbasis(hotoscdim), gellmannbasis(hotoscdim))...]])
      # 3) inserisco il vettore in un tensore con gli Index degli oscillatori
      iv = itensor(v, sites[1], sites[2])
      # 4) lo decompongo in due pezzi con una SVD
      f1, f2, _, _ = factorize(iv, sites[1]; which_decomp="svd")
      # 5) rinomino il Link tra i due fattori come "Link,l=1" anziché
      #    "Link,fact" che è il Tag assegnato da `factorize`
      replacetags!(f1, "fact" => "l=1")
      replacetags!(f2, "fact" => "l=1")

      ρ₀ = chain(MPS([f1, f2]),
                 parse_init_state(sites[range_spins],
                                  parameters["chain_initial_state"]),
                 parse_init_state_osc(sites[end], "empty"))
    elseif parameters["left_oscillator_initial_state"] == "empty"
      ρ₀ = chain(parse_init_state_osc(sites[1], "empty"),
                 parse_init_state_osc(sites[2], "empty"),
                 parse_init_state(sites[range_spins],
                                  parameters["chain_initial_state"]),
                 parse_init_state_osc(sites[end], "empty"))
    end

    # Osservabili
    # -----------
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)
    current_adjsites(ρ) = real.([inner(j, ρ)
                                 for j ∈ current_adjsites_ops]) ./ trace(ρ)
    osclevelsL1(ρ) = real.(inner.(Ref(ρ), osclevels_projs_L1)) ./ trace(ρ)
    osclevelsL2(ρ) = real.(inner.(Ref(ρ), osclevels_projs_L2)) ./ trace(ρ)

    # Ricorda che la numerazione degli oscillatori a sx è invertita: b₁ sta
    # al posto 2, e b₂ sta al posto 1
    b2⁺b1 = MPS(sites, ["veca+"; "veca-"; repeat(["vecId"], length(sites)-2)])
    b1⁺b2 = MPS(sites, ["veca-"; "veca+"; repeat(["vecId"], length(sites)-2)])
    function originaloccn(ρ)
      trρ = trace(ρ)
      avgb1⁺b1 = inner(num_op_list[2], ρ) / trρ
      avgb2⁺b2 = inner(num_op_list[1], ρ) / trρ
      avgb2⁺b1 = inner(b2⁺b1, ρ) ./ trρ
      avgb1⁺b2 = inner(b1⁺b2, ρ) ./ trρ
      cosθ = κ₁ / hypot(κ₁, κ₂)
      sinθ = κ₂ / hypot(κ₁, κ₂)
      return [real(cosθ^2 * avgb1⁺b1 + sinθ^2 * avgb2⁺b2 -
                   sinθ*cosθ * (avgb2⁺b1+avgb1⁺b2)),
              real(sinθ^2 * avgb1⁺b1 + cosθ^2 * avgb2⁺b2 +
                   sinθ*cosθ * (avgb2⁺b1+avgb1⁺b2))]
              # Il primo è il # di occ dell'oscillatore con il picco a Ω,
              # l'altro a -Ω. Impongo `real` perché so che il risultato deve
              # essere un numero reale.
    end

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout,
      normalisation,
      occnlist,
      current_adjsites_list,
      ranks,
      osclevelsL1list,
      osclevelsL2list,
      originaloccnlist = evolve(ρ₀,
                                time_step_list,
                                parameters["skip_steps"],
                                parameters["TS_expansion_order"],
                                links_odd,
                                links_even,
                                parameters["MP_compression_error"],
                                parameters["MP_maximum_bond_dimension"];
                                fout=[trace,
                                      occn,
                                      current_adjsites,
                                      linkdims,
                                      osclevelsL1,
                                      osclevelsL2,
                                      originaloccn])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    osclevelsL1list = mapreduce(permutedims, vcat, osclevelsL1list)
    osclevelsL2list = mapreduce(permutedims, vcat, osclevelsL2list)
    current_adjsites_list = mapreduce(permutedims, vcat, current_adjsites_list)
    ranks = mapreduce(permutedims, vcat, ranks)

    n⁺, n⁻ = originaloccnlist[end]
    dbratio = (1+n⁺) / n⁻
    dbrange = (dbratio-0.5, dbratio+0.5)
    originaloccnlist = mapreduce(permutedims, vcat, originaloccnlist)

    @info "($current_sim_n di $tot_sim_n) Creazione delle tabelle di output."
    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    sitelabels = ["L2"; "L1"; string.("S", 1:n_spin_sites); "R"]
    for (j, label) ∈ enumerate(sitelabels)
      sym = Symbol(string("occn_", label))
      push!(dict, sym => occnlist[:,j])
    end
    for coln ∈ eachindex(current_adjsites_ops)
      s = Symbol("current_S$coln/S$(coln+1)")
      push!(dict, s => current_adjsites_list[:, coln])
    end
    for j ∈ eachindex(sitelabels)[1:end-1]
      s = Symbol("bond_dim$(sitelabels[j])/$(sitelabels[j+1])")
      push!(dict, s => ranks[:, j])
    end
    push!(dict, :orig_occn_1 =>  originaloccnlist[:, 1])
    push!(dict, :orig_occn_2 =>  originaloccnlist[:, 2])
    push!(dict, :dbratio => (1 .+ originaloccnlist[:, 1]) ./ originaloccnlist[:, 2])
    push!(dict, :trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occn_super, occnlist)
    push!(originaloccn_super, originaloccnlist)
    push!(originaloccn_yrange_super, dbrange)
    push!(current_adjsites_super, current_adjsites_list)
    push!(bond_dimensions_super, ranks)
    push!(normalisation_super, normalisation)
    push!(osclevelsL1_super, osclevelsL1list)
    push!(osclevelsL2_super, osclevelsL2list)
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
                           occ_n_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c, ls) ∈ zip(eachcol(data),
                           readablecolours(N),
                           ["dashed";
                            "dashed";
                            repeat(["solid"], N-3);
                            "dashed"])
        plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
        push!(ax, plot)
      end
      push!(ax, Legend( ["L1"; "L2"; string.(1:N-3); "R"] ))
      push!(grp, ax)
    end
    pgfsave("populations.pdf", grp)
  end

  # Population of spin sites
  spinsonly = [mat[:, 3:end-1] for mat ∈ occ_n_super]
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
      push!(ax, Legend( string.(1:N) ))
      push!(grp, ax)
    end
    pgfsave("population_spin_sites.pdf", grp)
  end

  # Grafico dei numeri di occupazione pre-trasformazione
  # ----------------------------------------------------
  data = [[d[:, 1] d[:, 2] (1 .+ d[:, 1])./d[:, 2]] for d ∈ originaloccn_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
    })
    for (t, data, p, rn, ls) ∈ zip(timesteps_super,
                                   data,
                                   parameter_lists,
                                   originaloccn_yrange_super,
                                   ["solid", "solid", "dashed"])
      ax = Axis({title = filenamett(p), ymin = rn[1], ymax = rn[2]})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( [L"n(\Omega)",
                         L"n(-\Omega)",
                         L"[1+n(\Omega)]/n(-\Omega)"] ))
      push!(grp, ax)
    end
    pgfsave("population_original_oscillators.pdf", grp)
  end

  # Occupation numbers (oscillators + chain total)
  #
  # sum(X, dims=2) prende la matrice X e restituisce un vettore colonna
  # le cui righe sono le somme dei valori sulle rispettive righe di X.
  sums = [hcat(sum(mat[:, 1:2], dims=2),
               sum(mat[:, 3:end-1], dims=2),
               mat[:, end])
          for mat ∈ occn_super]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle n_i(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           sums,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2)
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( ["L1+L2", "spin total", "R"] ))
      push!(grp, ax)
    end
    pgfsave("population_sums.pdf", grp)
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
      N = size(data, 2) # = no. of spin sites + 2
      sitelabels = ["L2"; "L1"; string.(1:N-2); "R"]
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(sitelabels) ))
      push!(grp, ax)
    end
    pgfsave("bond_dimensions.pdf", grp)
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
                             normalisation_super,
                             parameter_lists,
                             readablecolours(length(parameter_lists)))
      plot = PlotInc({color = col}, Table([t, y]))
      push!(ax, plot)
      push!(ax, LegendEntry(filenamett(p)))
    end
    pgfsave("normalisation.pdf", ax)
  end


  # Particle current, between adjacent sites
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle j_{i,i+1}(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           current_adjsites_super,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2) + 1 # N ≡ no. of spin sites
      for (y, c) ∈ zip(eachcol(data), readablecolours(N-1))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(1:N) ))
      push!(grp, ax)
    end
    pgfsave("particle_current_adjacent_sites.pdf", grp)
  end

  # Occupations of the number operator eigenspaces in the pseudomodes
  for (list, pos) in zip([osclevelsL2_super, osclevelsL1_super],
                         ["L2", "L1"])
    @pgf begin
      grp = GroupPlot({
                       group_opts...,
                       xlabel = L"\lambda t",
                       ylabel = L"n",
                      })
      for (t, data, p) ∈ zip(timesteps_super,
                             list,
                             parameter_lists)
        ax = Axis({title = filenamett(p)})
        N = size(data, 2) - 1
        for (y, c, ls) ∈ zip(eachcol(data),
                             readablecolours(N+1),
                             [repeat(["solid"], N); "dashed"])
          plot = Plot({ color = c }, Table([t, y]))
 plot[ls] = nothing
          push!(ax, plot)
        end
        push!(ax, Legend( [string.(0:N-1); "total"] ))
        push!(grp, ax)
      end
      pgfsave("$(site)_pmode_number_eigenspaces.pdf", grp)
    end
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
