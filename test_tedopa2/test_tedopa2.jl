#!/usr/bin/julia

using ITensors
using LaTeXStrings
using ProgressMeter
using Base.Filesystem
using DataFrames
using CSV

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/tedopa.jl")
include(lib_path * "/operators_pytedopa.jl")

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
  occ_n_super = []
  bond_dimensions_super = []
  osc_chain_coefficients_super = []
  snapshot_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri tecnici
    max_err = parameters["MP_compression_error"]
    max_dim = parameters["MP_maximum_bond_dimension"]
    nquad = Int(parameters["PolyChaos_nquad"])

    # - parametri fisici
    ε1 = parameters["spin_excitation_energy"]
    ε2 = 0.0
    K = parameters["spin_interaction_energy"]
    g = parameters["frequency_cutoff"]
    osc_dim = parameters["oscillator_space_dimension"]
    λ = parameters["spectral_density_prop_constant"]
    γ = parameters["spectral_density_half_width"]

    # - intervallo temporale delle simulazioni
    τ = parameters["simulation_time_step"]
    time_step_list = construct_step_list(parameters)
    skip_steps = parameters["skip_steps"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]
    n_osc_left = parameters["number_of_oscillators_left"]
    n_osc_right = parameters["number_of_oscillators_right"]
    sites = [siteinds("Osc", n_osc_left; dim=osc_dim);
             siteinds("S=1/2", n_spin_sites);
             siteinds("Osc", n_osc_right; dim=osc_dim)]

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
    @info "Calcolo dei parametri TEDOPA in corso."
    support = (0, g)
    J(x) = 8λ * γ * x / (x^2 + γ^2)

    (Ωₗ, κₗ, ηₗ) = chainmapcoefficients(J,
                                        support,
                                        g,
                                        n_osc_left-1;
                                        Nquad=nquad,
                                        discretization=lanczos)
    (Ωᵣ, κᵣ, ηᵣ) = (Ωₗ, κₗ, ηₗ)
    # In caso di bisogno, le righe seguenti prendono i coefficienti
    # di ricorsione calcolati da py-tedopa per la funzione spettrale
    # J(x) = 8λ * γ * x / (x^2 + γ^2)
    #@assert n_osc_left == n_osc_left
    #α = parse.(Float64, readlines("alphas_lorentz.dat"))
    #β = parse.(Float64, readlines("betas_lorentz.dat"))
    #Ω = g .* α[1:n_osc_right]
    #κ = g .* sqrt.(β[2:n_osc_right])
    #η = sqrt(β[1])
    #(Ωₗ, κₗ, ηₗ) = (Ω, κ, η)
    #(Ωᵣ, κᵣ, ηᵣ) = (Ω, κ, η)

    # Raccolgo i coefficienti in array per poterli disegnare assieme nei
    # grafici. (I coefficienti κ sono uno in meno degli Ω! Per ora
    # pareggio le lunghezze inserendoci η all'inizio...)
    osc_chain_coefficients = [Ωₗ [ηₗ; κₗ] Ωᵣ [ηᵣ; κᵣ]]

    localcfs = [reverse(Ωₗ); [ε1, ε2]; Ωᵣ]
    interactioncfs = [reverse(κₗ); [ηₗ, K, ηᵣ]; κᵣ]
    hlist = twositeoperators(sites, localcfs, interactioncfs)
    #
    function links_odd(τ)
      return [exp(-im * τ * h) for h in hlist[1:2:end]]
    end
    function links_even(τ)
      return [exp(-im * τ * h) for h in hlist[2:2:end]]
    end
    #
    evo = evolution_operator(links_odd,
                             links_even,
                             τ,
                             parameters["TS_expansion_order"])

    # Simulazione
    # ===========
    # Stato iniziale
    # --------------
    # Gli oscillatori partono tutti dallo stato vuoto
    current_state = chain(MPS(sites[range_osc_left], "0"),
                          MPS(sites[range_spins], ["Up", "Dn"]),
                          MPS(sites[range_osc_right], "0"))

    # Osservabili sullo stato iniziale
    # --------------------------------
    occ_n = [expect(current_state, "N")]
    bond_dimensions = [linkdims(current_state)]

    # Evoluzione temporale
    # --------------------
    @info "Avvio della simulazione."
    message = "Simulazione $current_sim_n di $tot_sim_n:"
    progress = Progress(length(time_step_list), 1, message, 30)
    skip_count = 1

    for _ in time_step_list[2:end]
      current_state = apply(evo,
                            current_state;
                            cutoff=max_err,
                            maxdim=max_dim)
      if skip_count % skip_steps == 0
        push!(occ_n, expect(current_state, "N"))
        push!(bond_dimensions, linkdims(current_state))
      end
      next!(progress)
      skip_count += 1
    end

    snapshot = occ_n[end]

    ## Creo una tabella con i dati rilevanti da scrivere nel file di output
    #dict = Dict(:time => time_step_list[1:skip_steps:end])
    #tmp_list = hcat(occ_n...)
    #for (j, name) in enumerate([[Symbol("occ_n_left$n") for n∈n_osc_left:-1:1];
    #                            [Symbol("occ_n_spin$n") for n∈1:n_spin_sites];
    #                            [Symbol("occ_n_right$n") for n∈1:n_osc_right]])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(spin_current...)
    #for (j, name) in enumerate([Symbol("spin_current$n")
    #                            for n = 1:n_spin_sites-1])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(spin_chain_levels...)
    #for (j, name) in enumerate([Symbol("levels_chain$n") for n = 0:n_spin_sites])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #tmp_list = hcat(bond_dimensions...)
    #len = n_osc_left + n_spin_sites + n_osc_right
    #for (j, name) in enumerate([Symbol("bond_dim$n") for n ∈ 1:len-1])
    #  push!(dict, name => tmp_list[j,:])
    #end
    #table = DataFrame(dict)
    #filename = replace(parameters["filename"], ".json" => "") * ".dat"
    ## Scrive la tabella su un file che ha la stessa estensione del file dei
    ## parametri, con estensione modificata.
    #CSV.write(filename, table)

    # Salvo i risultati nei grandi contenitori
    push!(occ_n_super, occ_n)
    push!(bond_dimensions_super, bond_dimensions)
    push!(osc_chain_coefficients_super, osc_chain_coefficients)
    push!(snapshot_super, snapshot)
  end

  # Grafici
  # =======
  plot_size = (2, 0.5 + ceil(length(parameter_lists)/2)) .* (600, 400)

  distinct_p, repeated_p = categorise_parameters(parameter_lists)

  # Grafico dei numeri di occupazione
  # ---------------------------------------------
  len = size(hcat(occ_n_super[begin]...), 1)
  plt = plot_time_series(occ_n_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=repeat([nothing], len),
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"\langle n_i\rangle",
                         plot_title="Numeri di occupazione",
                         plot_size=plot_size)
  savefig(plt, "occ_n.png")

  # Grafico dei ranghi del MPS
  # --------------------------
  len = size(hcat(bond_dimensions_super[begin]...), 1)
  plt = plot_time_series(bond_dimensions_super,
                         parameter_lists;
                         displayed_sites=nothing,
                         labels=repeat([nothing], len),
                         linestyles=repeat([:solid], len),
                         x_label=L"\lambda\, t",
                         y_label=L"\chi_{k,k+1}",
                         plot_title="Ranghi del MPS",
                         plot_size=plot_size
                        )
  savefig(plt, "bond_dimensions.png")
 
  # Grafico dei coefficienti della chain map
  # ----------------------------------------
  plt = plot_standalone(osc_chain_coefficients_super,
                        parameter_lists;
                        labels=[L"\Omega^{(L)}_i",
                                L"\kappa^{(L)}_i",
                                L"\Omega^{(R)}_i",
                                L"\kappa^{(R)}_i"],
                        linestyles=repeat([:solid], 4),
                        x_label=L"i",
                        y_label="Coefficiente",
                        plot_title="Coefficienti della catena di "*
                                   "oscillatori (sx e dx)",
                        plot_size=plot_size
                        )
  savefig(plt, "osc_coefficients.png")

  # Istantanea dei numeri di occupazione alla fine
  # ----------------------------------------------
  plt = plot_standalone(snapshot_super,
                        parameter_lists;
                        labels=[nothing],
                        linestyles=[:solid],
                        x_label=L"i",
                        y_label="Numero di occupazione",
                        plot_title="Numeri di occupazione alla fine",
                        plot_size=plot_size
                        )
  savefig(plt, "snapshot.png")

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
