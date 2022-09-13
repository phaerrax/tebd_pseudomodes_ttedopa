#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using ITensorTDVP, Observers, PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma calcola l'evoluzione della catena di spin isolata,
# usando le tecniche dei MPS ed MPO.

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
    entropy_super = []
    current_allsites_super = []
    timesteps_super = []

    for (current_sim_n, parameters) in enumerate(parameter_lists)
        # - parametri fisici
        ε = parameters["spin_excitation_energy"]
        # λ = 1

        # - intervallo temporale delle simulazioni
        timestep = parameters["simulation_time_step"]
        endtime = parameters["simulation_end_time"]

        # Costruzione della catena
        # ========================
        n_sites = parameters["number_of_spin_sites"] # per ora deve essere un numero pari
        # L'elemento site[i] è l'Index che si riferisce al sito i-esimo
        sites = siteinds("S=1/2", n_sites)

        # Costruzione dell'operatore di evoluzione
        # ========================================
        function hamiltonian_xy(N, energy, coupling)
            op = OpSum()
            for j in 1:(N-1)
                op += -0.5coupling, "S+", j, "S-", j + 1
                op += -0.5coupling, "S-", j, "S+", j + 1
            end
            for j in 1:N
                op += energy, "Sz", j
            end
            return op
        end
        H = MPO(hamiltonian_xy(n_sites, ε, 1), sites)

        # Simulazione
        # ===========
        # Determina lo stato iniziale a partire dalla stringa data nei parametri
        ψ₀ = parse_init_state(sites, parameters["chain_initial_state"])

        # Misuro le osservabili sullo stato iniziale
        current_allsites_ops = [[fermioncurrent(sites, s, j)
                                 for j ∈ filter(n -> n != s, 1:n_sites)]
                                for s ∈ 1:n_sites]

        function occn(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                return real.(expect(psi, "N"))
            end
            return nothing
        end
        function entropy(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                return real.([vonneumannentropy(psi, sites, j) for j in 2:n_sites])
            end
            return nothing
        end
        function spincurrent(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                return reduce(vcat,
                              [real.([inner(psi', j, psi) for j in ops])
                               for ops ∈ current_allsites_ops])
            end
            return nothing
        end
        function currenttime(; current_time, bond, half_sweep)
            # Get the times at which the observable are computed.
            if bond == 1 && half_sweep == 2
                return -imag(current_time)
                # The TDVP is run with imaginary time steps (look below).
            end
            return nothing
        end
        function bonddimensions(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                return linkdims(psi)
            end
            return nothing
        end

        obs = Observer(
            "times" => currenttime,
            "occn" => occn,
            "entropy" => entropy,
            "current" => spincurrent,
            "ranks" => bonddimensions,
        )

        @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

        ψ_f = tdvp(
            H,
            -im * endtime,
            ψ₀;
            time_step=-im * timestep,
            normalize=false,
            (observer!)=obs,
            cutoff=parameters["MP_compression_error"],
            mindim=parameters["MP_minimum_bond_dimension"],
            maxdim=parameters["MP_maximum_bond_dimension"],
        )

        function groupresults(obs::Observer, name::String)
            return mapreduce(permutedims, vcat, results(obs, name))
        end

        # A partire dai risultati costruisco delle matrici da dare poi in pasto
        # alle funzioni per i grafici e le tabelle di output
        tout = results(obs, "times")
        occnlist = groupresults(obs, "occn")
        entropylist = groupresults(obs, "entropy")
        currentlist = groupresults(obs, "current")
        ranks = groupresults(obs, "ranks")

        push!(timesteps_super, tout)
        push!(occ_n_super, occnlist)
        push!(bond_dimensions_super, ranks)
        push!(current_allsites_super, currentlist)
        push!(entropy_super, entropylist)
    end

    # Plots
    # -----
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
            for (y, c) ∈ zip(eachcol(data), readablecolours(N))
                plot = Plot({ color = c }, Table([t, y]))
                push!(ax, plot)
            end
            push!(ax, Legend( string.(1:N) ))
            push!(grp, ax)
        end
        pgfsave("occ_n.pdf", grp)
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
            for (y, c) ∈ zip(eachcol(data), readablecolours(N))
                plot = Plot({ color = c }, Table([t, y]))
                push!(ax, plot)
            end
            push!(ax, Legend( ["($j,$(j+1))" for j ∈ 1:N] ))
            push!(grp, ax)
        end
        pgfsave("bond_dimensions.pdf", grp)
    end

    # Entanglement entropy
    @pgf begin
        grp = GroupPlot({
                         group_opts...,
                         xlabel = L"\lambda t",
                         ylabel = L"S_i(t)",
                        })
        for (t, data, p) ∈ zip(timesteps_super,
                               entropy_super,
                               parameter_lists)
            ax = Axis({title = filenamett(p)})
            N = size(data, 2)
            for (y, c) ∈ zip(eachcol(data), readablecolours(N))
                plot = Plot({ color = c }, Table([t, y]))
                push!(ax, plot)
            end
            push!(ax, Legend( string.(1 .+ (1:N)) ))
            push!(grp, ax)
        end
    end
    pgfsave("entropy.pdf", grp)

    cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
    return
end
