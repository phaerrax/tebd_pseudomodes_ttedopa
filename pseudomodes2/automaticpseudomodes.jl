#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, QuadGK, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma prende la densità spettrale antisimmetrica lorentziana e,
# seguendo quanto scritto nell'articolo di Mascherpa e altri, la mappa (al
# meglio possibile) in una coppia di pseudomodi a temperatura nulla.

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
  osc_dim_list = [p["oscillator_space_dimension"]
                  for p ∈ parameter_lists]
  if allequal(n_spin_sites_list) && allequal(osc_dim_list)
    preload = true
    n_spin_sites = first(n_spin_sites_list)
    osc_dim = first(osc_dim_list)

    spin_range = 2 .+ (1:n_spin_sites)

    sites = [siteinds("HvOsc", 2; dim=osc_dim);
             siteinds("HvS=1/2", n_spin_sites);
             siteinds("HvOsc", 1; dim=osc_dim)]

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
    full_trace = MPS(sites, "vecId")
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
    Ω = parameters["spectral_density_peak"]
    γ = parameters["spectral_density_half_width"]
    κ = parameters["spectral_density_overall_factor"]
    T = parameters["temperature"]
    # La densità spettrale:
    J(ω) = κ^2/π * (0.5γ / ((0.5γ)^2 + (ω-Ω)^2) - 
                    0.5γ / ((0.5γ)^2 + (ω+Ω)^2))
    # La funzione di correlazione:
    # c(t) = ∫ J(ω) [ exp(-iωt) (1+n(T,ω)) + exp(iωt) n(T,ω) ] dω, su (0,+∞)
    n(T,ω) = T == 0 ? 0 : (ℯ^(ω/T) - 1)^(-1)
    correlationf(t) = quadgk(ω -> J(ω) * (ℯ^(-im*ω*t)*(1+n(T,ω)) +
                                          ℯ^(im*ω*t)*n(T,ω) ),
                             0, Inf)[begin]

    # Calcolo i parametri degli pseudomodi:
    w, λ = prony(correlationf, 0, 2, 0.1, 2)

    # Dall'equazione (B6) nell'articolo:
    α₁ = (w[1]*λ[1] + w[2]*λ[2]) / (w[1] + w[2])
    α₂ = (w[1]*λ[2] + w[2]*λ[1]) / (w[1] + w[2])
    η² = 1/4*(λ[1] - λ[2])^2 * ( ( (w[1]-w[2])/(w[1]+w[2]) )^2 - 1 )
    @assert abs(imag(η²)) ≤ 1e-10
    @assert abs(imag(w[1] + w[2])) ≤ 1e-10
    η = sqrt(real(η²))
    κ₁ = real(w[1] + w[2])
    κ₂ = 0
    # Dalle α ottengo γ e ω:
    ω₁ = -imag(α₁)
    γ₁ = -2real(α₁)
    ω₂ = -imag(α₂) 
    γ₂ = -2real(α₂) 
    # Per l'oscillatore a destra uso ancora il vecchio metodo:
    ωᵣ = Ω
    κᵣ = κ
    γᵣ = γ

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    if !preload
      n_spin_sites = parameters["number_of_spin_sites"]
      spin_range = 2 .+ (1:n_spin_sites)

      sites = [siteinds("HvOsc", 2; dim=osc_dim);
               siteinds("HvS=1/2", n_spin_sites);
               siteinds("HvOsc", 1; dim=osc_dim)]
    end

    # Definizione degli operatori nell'equazione di Lindblad
    # ======================================================
    # Calcolo i coefficienti dell'Hamiltoniano trasformato.
    # Tutto è a temperatura nulla, quindi esplicito T=0 dappertutto.
    ω̃₁ = (ω₁*κ₁^2 + ω₂*κ₂^2 + 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)
    ω̃₂ = (ω₂*κ₁^2 + ω₁*κ₂^2 - 2η*κ₁*κ₂) / (κ₁^2 + κ₂^2)
    κ̃₁ = sqrt(κ₁^2 + κ₂^2)
    κ̃₂ = ((ω₂-ω₁)*κ₁*κ₂ + η*(κ₁^2 - κ₂^2)) / (κ₁^2 + κ₂^2)
    #γ̃₁⁺ = (κ₁^2*γ₁*(n(T,ω₁)+1) + κ₂^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    #γ̃₁⁻ = (κ₁^2*γ₁* n(T,ω₁)    + κ₂^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)
    #γ̃₂⁺ = (κ₂^2*γ₁*(n(T,ω₁)+1) + κ₁^2*γ₂*(n(T,ω₂)+1)) / (κ₁^2 + κ₂^2)
    #γ̃₂⁻ = (κ₂^2*γ₁* n(T,ω₁)    + κ₁^2*γ₂* n(T,ω₂))    / (κ₁^2 + κ₂^2)
    #γ̃₁₂⁺ = κ₁*κ₂*( γ₂*(n(T,ω₂)+1) - γ₁*(n(T,ω₁)+1) ) / (κ₁^2 + κ₂^2)
    #γ̃₁₂⁻ = κ₁*κ₂*( γ₂*n(T,ω₂)     - γ₁*n(T,ω₁) )     / (κ₁^2 + κ₂^2)
    γ̃₁⁺ = (κ₁^2*γ₁ + κ₂^2*γ₂) / (κ₁^2 + κ₂^2)
    γ̃₁⁻ = 0
    γ̃₂⁺ = (κ₂^2*γ₁ + κ₁^2*γ₂) / (κ₁^2 + κ₂^2)
    γ̃₂⁻ = 0
    γ̃₁₂⁺ = κ₁*κ₂*(γ₂-γ₁) / (κ₁^2 + κ₂^2)
    γ̃₁₂⁻ = 0
    localcfs = [ω̃₂; ω̃₁; repeat([ε], n_spin_sites); ωᵣ]
    interactioncfs = [κ̃₂; κ̃₁; repeat([1], n_spin_sites-1); κᵣ]
    ℓlist = twositeoperators(sites, localcfs, interactioncfs)
    # Aggiungo agli operatori già creati gli operatori di dissipazione (con T=0):
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
    #ℓlist[1] += (γ̃₁₂⁺ * mixedlindbladplus(sites[1], sites[2]) +
    #             γ̃₁₂⁻ * mixedlindbladminus(sites[1], sites[2]))
    ℓlist[1] += γ̃₁₂⁺ * mixedlindbladplus(sites[1], sites[2])
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
      full_trace = MPS(sites, "vecId")
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
    # Non c'è scelta qui, lo stato iniziale è per forza a temperatura nulla.
    ρ₀ = chain(parse_init_state_osc(sites[1], "empty"),
               parse_init_state_osc(sites[2], "empty"),
               parse_init_state(sites[spin_range],
                                parameters["chain_initial_state"]),
               parse_init_state_osc(sites[end], "empty"))

    # Osservabili
    # -----------
    trace(ρ) = real(inner(full_trace, ρ))
    occn(ρ) = real.([inner(N, ρ) / trace(ρ) for N in num_op_list])
    current_adjsites(ρ) = real.([inner(j, ρ) / trace(ρ)
                                 for j ∈ current_adjsites_ops])
    function current_allsites(ρ)
      pairs = [(i,j) for i ∈ 1:n_spin_sites for j ∈ 1:n_spin_sites if j > i]
      mat = zeros(n_spin_sites, n_spin_sites)
      for (j, i) in zip(current_allsites_ops, pairs)
        mat[i...] = real(inner(j, ρ) / trace(ρ))
      end
      mat .-= transpose(mat)
      return Base.vec(mat')
    end

    # Evoluzione temporale
    # --------------------
    @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

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
    push!(dict, :full_trace => normalisation)
    table = DataFrame(dict)
    filename = replace(parameters["filename"], ".json" => "") * ".dat"
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
    legend_pos = "outer north east"
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
               "legend pos" = "outer north east"
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
      push!(ax, Legend( string.([1:N]) ))
      push!(grp, ax)
    end
    pgfsave("particle_current_from_1st_spin.pdf", grp)
  end

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
