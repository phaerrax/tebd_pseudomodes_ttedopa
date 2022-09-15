#!/usr/bin/julia

using ITensors, LaTeXStrings, DataFrames, CSV, PGFPlotsX, Colors
using ITensorTDVP, Observers, PseudomodesTTEDOPA

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
    occ_n_super = []
    current_adjsites_super = []
    bond_dimensions_super = []
    chain_levels_super = []
    normalisation_super = []

    for (current_sim_n, parameters) in enumerate(parameter_lists)
        # Costruzione della catena
        # ========================
        ε = parameters["spin_excitation_energy"]
        κ = parameters["spin_damping_coefficient"]
        T = parameters["temperature"]
        n_sites = parameters["number_of_spin_sites"]
        sites = siteinds("HvS=1/2", n_sites)

        # Costruzione dell'operatore di evoluzione
        # ========================================
        avgn(ε, T) = T == 0 ? 0 : (ℯ^(ε / T) - 1)^(-1)
        function lindbladian_xy(
            chain_length::Integer,
            spin_energy::Real,
            spin_coupling::Real
        )
            N = chain_length
            ε = spin_energy
            λ = spin_coupling

            op = OpSum()
            for j in 1:(N - 1)
                op += +0.5im * λ, "σ-⋅", j, "σ+⋅", j + 1
                op += +0.5im * λ, "σ+⋅", j, "σ-⋅", j + 1
                op += -0.5im * λ, "⋅σ-", j, "⋅σ+", j + 1
                op += -0.5im * λ, "⋅σ+", j, "⋅σ-", j + 1
            end
            for j in 1:N
                op += -0.5im * ε, "σz⋅", j
                op += +0.5im * ε, "⋅σz", j
            end
            return op
        end
        function dissipator_symmetric(
            n::Integer,
            excitation_energy::Real,
            spin_damping_coefficient::Real,
            temperature::Real,
        )
            ε = excitation_energy
            κ = spin_damping_coefficient
            T = temperature
            ξ = κ * (1 + 2avgn(ε, T))

            op = OpSum()
            op += +ξ, "σx⋅ * ⋅σx", n
            op += -ξ, "Id", n
            return op
        end

       L = (
           MPO(lindbladian_xy(n_sites, ε, 1), sites) +
           MPO(dissipator_symmetric(1, ε, κ, 0), sites) +
           MPO(dissipator_symmetric(n_sites, ε, κ, T), sites)
       )

        # Osservabili da misurare
        # =======================
        full_trace = MPS(sites, "vecId")

        current_adjsitesops = [
            fermioncurrent(sites, j, j + 1) for j in eachindex(sites)[1:(end - 1)]
        ]

        numlabels = [[i == n ? "vecN" : "vecId" for i in 1:n_sites] for n in 1:n_sites]
        num_op_list = MPS.(Ref(sites), numlabels)

        function occn(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                tr = real(inner(full_trace, psi))
                return real.(inner.(num_op_list, Ref(psi))) ./ tr
            end
            return nothing
        end
        function spincurrent(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                tr = real(inner(full_trace, psi))
                return real.(inner.(current_adjsitesops, Ref(psi))) ./ tr
            end
            return nothing
        end
        function currenttime(; current_time, bond, half_sweep)
            # Get the times at which the observable are computed.
            if bond == 1 && half_sweep == 2
                return real(current_time)
                # See note below (before `tdvp` is called) on why `real` is used here.
            end
            return nothing
        end
        function bonddimensions(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                return linkdims(psi)
            end
            return nothing
        end
        function normalization(; psi, bond, half_sweep)
            if bond == 1 && half_sweep == 2
                return real(inner(full_trace, psi))
            end
            return nothing
        end

        # Simulazione
        # ===========
        # Lo stato iniziale della catena è dato da "chain_initial_state".
        ρ₀ = parse_init_state(sites, parameters["chain_initial_state"])

        # Evoluzione temporale
        # --------------------
        @info "($current_sim_n di $tot_sim_n) Avvio della simulazione."

        obs = Observer(
            "norm"    => normalization,
            "times"   => currenttime,
            "occn"    => occn,
            "current" => spincurrent,
            "ranks"   => bonddimensions,
        )

        # In the code that executes the evolution of a block of the MPS in the TDVP
        # method, a certain matrix is created, using as eltype the "promoted type"
        # between ρ₀'s type and the time step's type. Both here would normally be
        # Float64, so this matrix is Float64 too.
        # [ITensorTDVP/applyexp.jl, line 35]
        # The problem is that L is built from ComplexF64 matrices, therefore it
        # has a ComplexF64 type, even though mathematically speaking it is real;
        # in practice, its components have negligible (but nonzero) imaginary parts.
        # This causes a ComplexF64 number, instead of a Float64 one, to be inserted
        # in the aforementioned matrix, which raises an InexactError.
        #
        # The solution? Use a complex time step with a null imaginary part.
        ρ_f = tdvp(
            L,
            0im + parameters["simulation_end_time"],
            ρ₀;
            time_step=0im + parameters["simulation_time_step"],
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
        tout        = results(obs, "times")
        statenorm   = results(obs, "norm")
        occnlist    = groupresults(obs, "occn")
        currentlist = groupresults(obs, "current")
        ranks       = groupresults(obs, "ranks")

        # Creo una tabella con i dati rilevanti da scrivere nel file di output
        dict = Dict(:time => tout)
        sitelabels = string.("S", 1:n_sites)
        for (n, label) in enumerate(sitelabels)
            push!(dict, Symbol(string("occn_", label)) => occnlist[:, n])
        end
        for n in 1:(n_sites - 1)
            from = sitelabels[n]
            to   = sitelabels[n + 1]
            sym  = "current_adjsites_$from/$to"
            push!(dict, Symbol(sym) => currentlist[:, n])
            sym = "rank_$from/$to"
            push!(dict, Symbol(sym) => ranks[:, n])
        end
        push!(dict, :norm => statenorm)
        table = DataFrame(dict)
        filename = replace(parameters["filename"], ".json" => ".dat")
        # Scrive la tabella su un file che ha la stessa estensione del file dei
        # parametri, con estensione modificata.
        CSV.write(filename, table)

        # Salvo i risultati nei grandi contenitori
        push!(timesteps_super, tout)
        push!(occ_n_super, occnlist)
        push!(current_adjsites_super, currentlist)
        push!(bond_dimensions_super, ranks)
        push!(normalisation_super, statenorm)
    end

    # Plots
    # -----
    # Common options for group plots
    nrows = Int(ceil(tot_sim_n / 2))
    common_opts = @pgf {
        no_markers,
        grid = "major",
        legend_pos = "outer north east",
        "every axis plot/.append style" = "thick",
    }
    group_opts = @pgf {
        group_style = {
            group_size        = "$nrows by 2",
            y_descriptions_at = "edge left",
            horizontal_sep    = "2cm",
            vertical_sep      = "2cm",
        },
        common_opts...,
    }

    # Occupation numbers
    @pgf begin
        grp = GroupPlot({
            group_opts..., xlabel = L"\lambda t", ylabel = L"\langle n_i(t)\rangle"
        })
        for (t, data, p) in zip(timesteps_super, occ_n_super, parameter_lists)
            ax = Axis({title = filenamett(p)})
            N = size(data, 2)
            for (y, c) in zip(eachcol(data), readablecolours(N))
                plot = Plot({color = c}, Table([t, y]))
                push!(ax, plot)
            end
            push!(ax, Legend(string.(1:N)))
            push!(grp, ax)
        end
        pgfsave("occ_n.pdf", grp)
    end

    # Bond dimensions
    @pgf begin
        grp = GroupPlot({group_opts..., xlabel = L"\lambda t", ylabel = L"\chi_{i,i+1}(t)"})
        for (t, data, p) in zip(timesteps_super, bond_dimensions_super, parameter_lists)
            ax = Axis({title = filenamett(p)})
            N = size(data, 2)
            for (y, c) in zip(eachcol(data), readablecolours(N))
                plot = Plot({color = c}, Table([t, y]))
                push!(ax, plot)
            end
            push!(ax, Legend(["($j,$(j+1))" for j in 1:N]))
            push!(grp, ax)
        end
        pgfsave("bond_dimensions.pdf", grp)
    end

    # Trace of the density matrix
    @pgf begin
        ax = Axis({
            xlabel = L"\lambda t",
            ylabel = L"\mathrm{tr}\rho(t)",
            title  = "Normalisation",
            common_opts...,
        })
        for (t, y, p, col) in zip(
            timesteps_super,
            normalisation_super,
            parameter_lists,
            readablecolours(length(parameter_lists)),
        )
            plot = PlotInc({color = col}, Table([t, y .- 1]))
            push!(ax, plot)
            push!(ax, LegendEntry(filenamett(p)))
        end
        pgfsave("normalisation.pdf", ax)
    end

    # Particle current
    @pgf begin
        grp = GroupPlot({
            group_opts..., xlabel = L"\lambda t", ylabel = L"\langle j_{i,i+1}(t)\rangle"
        })
        for (t, data, p) in zip(timesteps_super, current_adjsites_super, parameter_lists)
            ax = Axis({title = filenamett(p)})
            N = size(data, 2)
            for (y, c) in zip(eachcol(data), readablecolours(N))
                plot = Plot({color = c}, Table([t, y]))
                push!(ax, plot)
            end
            push!(ax, Legend(["($j,$(j+1))" for j in 1:N]))
            push!(grp, ax)
        end
        pgfsave("particle_current.pdf", grp)
    end

    cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
    return nothing
end
