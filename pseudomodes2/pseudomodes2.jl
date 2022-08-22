#!/usr/bin/julia

using LinearAlgebra, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using ITensors, PseudomodesTTEDOPA

const ⊗ = kron

disablegrifqtech()

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# In questo caso la catena è descritta dalla vettorizzazione della
# matrice densità, la quale evolve nel tempo secondo l'equazione
# di Lindblad.

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
  occ_n_super = []
  current_allsites_super = []
  current_adjsites_super = []
  bond_dimensions_super = []
  chain_levels_super = []
  osc_levels_left_super = []
  osc_levels_right_super = []
  normalisation_super = []

  # Precaricamento
  # ==============
  # Se in tutte le liste di parametri il numero di siti è lo stesso, posso
  # definire qui una volta per tutte alcuni elementi "pesanti" che servono dopo.
  n_spin_sites_list = [p["number_of_spin_sites"]
                       for p ∈ parameter_lists]
  # I file dei parametri potrebbero avere `cold_...` e `hot_...` oppure soltanto
  # `oscillator_space_dimension`. Devo differenziare i due casi.
  # Se cerco di costruire la lista dei `cold_...` ma nei file non c'è questa
  # chiave, Julia lancia un KeyError.
  # Eseguo quindi in questo punto del codice quello che faccio poi con γ:
  # controllo se esistono i due oscdim differenti, altrimenti prendo quello
  # in comune e lo copio nei due valori. Se non c'è neanche quello, va bene che
  # Julia lanci un errore: significa che i file di parametri sono malformati.
  for p ∈ parameter_lists
    if (!haskey(p, "hot_oscillator_space_dimension") &&
        !haskey(p, "cold_oscillator_space_dimension"))
      push!(p, "hot_oscillator_space_dimension" => p["oscillator_space_dimension"])
      push!(p, "cold_oscillator_space_dimension" => p["oscillator_space_dimension"])
    end
  end
  hotoscdim_list = [p["hot_oscillator_space_dimension"]
                    for p ∈ parameter_lists]
  coldoscdim_list = [p["cold_oscillator_space_dimension"]
                    for p ∈ parameter_lists]

  if (allequal(n_spin_sites_list) &&
      allequal(hotoscdim_list) &&
      allequal(coldscdim_list))
    preload = true
    n_spin_sites = first(n_spin_sites_list)
    hotoscdim = first(hotoscdim_list)
    coldscdim = first(coldscdim_list)

    spin_range = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=hotoscdim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=coldoscdim)]

    # - i numeri di occupazione
    num_op_list = [MPS(sites,
                       [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                   for n ∈ 1:length(sites)]

    # - la corrente tra siti
    current_allsites_ops = [fermioncurrent(sites, i, j)
                            for i ∈ spin_range
                            for j ∈ spin_range
                            if j > i]

    # - la normalizzazione (cioè la traccia) della matrice densità
    traceMPS = MPS(sites, "vecId")
  else
    preload = false
  end

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    @info "($current_sim_n di $tot_sim_n) Costruzione degli operatori di evoluzione temporale."
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
    hotoscdim = parameters["hot_oscillator_space_dimension"]
    coldoscdim = parameters["cold_oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    if !preload
      n_spin_sites = parameters["number_of_spin_sites"]
      spin_range = 2 .+ (1:n_spin_sites)

      sites = [siteinds("HvOsc", 2; dim=hotoscdim);
               siteinds("HvS=1/2", n_spin_sites);
               siteinds("HvOsc", 1; dim=coldoscdim)]
    end

    # Definizione degli operatori nell'equazione di Lindblad
    # ======================================================
    # Calcolo dei coefficienti dell'Hamiltoniano trasformato
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
    localcfs = [ω̃₂; ω̃₁; repeat([ε], n_spin_sites); ωᵣ]
    interactioncfs = [κ̃₂; κ̃₁; repeat([1], n_spin_sites-1); κᵣ]
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

    # Osservabili da misurare
    # =======================
    if !preload
      # - i numeri di occupazione
      num_op_list = [MPS(sites,
                         [i == n ? "vecN" : "vecId" for i ∈ 1:length(sites)])
                     for n ∈ 1:length(sites)]

      # - la corrente tra siti
      current_allsites_ops = [fermioncurrent(sites, i, j)
                              for i ∈ spin_range
                              for j ∈ spin_range
                              if j > i]

      # - la normalizzazione (cioè la traccia) della matrice densità
      traceMPS = MPS(sites, "vecId")
    end

    current_adjsites_ops = [-2κ̃₁*fermioncurrent(sites, 2, 3);
                            [fermioncurrent(sites, j, j+1)
                             for j ∈ spin_range[1:end-1]];
                            -2κᵣ*fermioncurrent(sites,
                                         spin_range[end],
                                         spin_range[end]+1)]

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
      v = vec(M, [êᵢ ⊗ êⱼ for (êᵢ, êⱼ) ∈ [Base.product(gellmannbasis(hotoscdim), gellmannbasis(hotoscdim))...]])
      # 3) inserisco il vettore in un tensore con gli Index degli oscillatori
      iv = itensor(v, sites[1], sites[2])
      # 4) lo decompongo in due pezzi con una SVD
      f1, f2, _, _ = factorize(iv, sites[1]; which_decomp="svd")
      # 5) rinomino il Link tra i due fattori come "Link,l=1" anziché
      #    "Link,fact" che è il Tag assegnato da `factorize`
      replacetags!(f1, "fact" => "l=1")
      replacetags!(f2, "fact" => "l=1")

      ρ₀ = chain(MPS([f1, f2]),
                 parse_init_state(sites[spin_range],
                                  parameters["chain_initial_state"]),
                 parse_init_state_osc(sites[end], "empty"))
    elseif parameters["left_oscillator_initial_state"] == "empty"
      ρ₀ = chain(parse_init_state_osc(sites[1], "empty"),
                 parse_init_state_osc(sites[2], "empty"),
                 parse_init_state(sites[spin_range],
                                  parameters["chain_initial_state"]),
                 parse_init_state_osc(sites[end], "empty"))
    end

    # Osservabili
    # -----------
    trace(ρ) = real(inner(traceMPS, ρ))
    occn(ρ) = real.([inner(N, ρ) for N in num_op_list]) ./ trace(ρ)
    current_adjsites(ρ) = real.([inner(j, ρ)
                                 for j ∈ current_adjsites_ops]) ./ trace(ρ)
    function current_allsites(ρ)
      pairs = [(i,j) for i ∈ 1:n_spin_sites for j ∈ 1:n_spin_sites if j > i]
      mat = zeros(n_spin_sites, n_spin_sites)
      trρ = trace(ρ)
      for (J, (k,l)) in zip(current_allsites_ops, pairs)
        mat[k,l] = real(inner(J, ρ) / trρ)
      end
      mat .-= transpose(mat)
      return Base.vec(mat')
    end

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

    @time begin
      tout,
      normalisation,
      occnlist,
      current_allsites_list,
      current_adjsites_list,
      ranks =  evolve(ρ₀,
                      time_step_list,
                      parameters["skip_steps"],
                      parameters["TS_expansion_order"],
                      links_odd,
                      links_even,
                      parameters["MP_compression_error"],
                      parameters["MP_maximum_bond_dimension"];
                      fout=[trace,
                            occn,
                            current_allsites,
                            current_adjsites,
                            linkdims])
    end

    # A partire dai risultati costruisco delle matrici da dare poi in pasto
    # alle funzioni per i grafici e le tabelle di output
    occnlist = mapreduce(permutedims, vcat, occnlist)
    current_allsites_list = mapreduce(permutedims, vcat, current_allsites_list)
    current_adjsites_list = mapreduce(permutedims, vcat, current_adjsites_list)
    ranks = mapreduce(permutedims, vcat, ranks)

    @info "($current_sim_n di $tot_sim_n) Creazione delle tabelle di output."
    # Creo una tabella con i dati rilevanti da scrivere nel file di output
    dict = Dict(:time => tout)
    for (j, name) in enumerate([:occ_n_left;
                                [Symbol("occ_n_spin$n") for n = 1:n_spin_sites];
                                :occ_n_right])
      push!(dict, name => occnlist[:,j])
    end
    syms = [Symbol("current_$i/$j")
            for i ∈ 1:n_spin_sites
            for j ∈ 1:n_spin_sites]
    for (coln, s) ∈ enumerate(syms)
        push!(dict, s => current_allsites_list[:, coln])
    end
    len = n_spin_sites + 2
    for (j, name) in enumerate([Symbol("bond_dim$n")
                                for n ∈ 1:len-1])
      push!(dict, name => ranks[:,j])
    end
    push!(dict, :trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => ".dat")
    # Scrive la tabella su un file che ha la stessa estensione del file dei
    # parametri, con estensione modificata.
    CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, tout)
    push!(occ_n_super, occnlist)
    push!(current_allsites_super, current_allsites_list)
    push!(current_adjsites_super, current_adjsites_list)
    push!(bond_dimensions_super, ranks)
    push!(normalisation_super, normalisation)
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
      push!(ax, Legend( ["L2" "L1"; string.(1:N-3); "R"] ))
      push!(grp, ax)
    end
    pgfsave("populations.pdf", grp)
  end

  # Population of spin sites
  spinsonly = [mat[:, 3:end-1] for mat in occ_n_super]
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

  # Occupation numbers (oscillators + chain total)
  #
  # sum(X, dims=2) prende la matrice X e restituisce un vettore colonna
  # le cui righe sono le somme dei valori sulle rispettive righe di X.
  sums = [hcat(sum(mat[:, 1:2], dims=2),
               sum(mat[:, 3:end-1], dims=2),
               mat[:, end])
          for mat ∈ occ_n_super]
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
      push!(ax, Legend( ["L1+L2" "spin total" "R"] ))
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
      N = size(data, 2)
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
      N = size(data, 2) # N ≡ no. of spin sites + 1
      sitelabels = ["L1"; string.(1:N-1); "R"]
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(sitelabels) ))
      push!(grp, ax)
    end
    pgfsave("particle_current_adjacent_sites.pdf", grp)
  end

  # Particle current, from the 1st site to the others
  data = [table[:, 1:p["number_of_spin_sites"]]
          for (p, table) in zip(parameter_lists, current_allsites_super)]
  @pgf begin
    grp = GroupPlot({
        group_opts...,
        xlabel = L"\lambda t",
        ylabel = L"\langle j_{1,i}(t)\rangle",
    })
    for (t, data, p) ∈ zip(timesteps_super,
                           data,
                           parameter_lists)
      ax = Axis({title = filenamett(p)})
      N = size(data, 2) # N ≡ no. of spin sites
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( string.(1:N) ))
      push!(grp, ax)
    end
    pgfsave("particle_current_from_1st_spin.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
