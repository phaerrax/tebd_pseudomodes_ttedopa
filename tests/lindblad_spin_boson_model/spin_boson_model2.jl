#!/usr/bin/julia

using ITensors, DataFrames, LaTeXStrings, CSV, PGFPlotsX, Colors
using PseudomodesTTEDOPA

disablegrifqtech()

# Questo programma calcola l'evoluzione della catena di spin
# smorzata agli estremi, usando le tecniche dei MPS ed MPO.
# Differisce da "chain_with_damped_oscillators" in quanto qui
# indago l'andamento di quantità diverse e lo confronto con una
# soluzione esatta (per un numero di spin sufficientemente basso).

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
  occ_n_MPS_super = []
  normalisation_super = []
  bond_dimensions_super = []
  occ_n_numeric_super = []
  diff_occ_n_super = []

  for (current_sim_n, parameters) in enumerate(parameter_lists)
    # Impostazione dei parametri
    # ==========================

    # - parametri fisici
    ε = parameters["spin_excitation_energy"]
    # λ = 1
    κ = parameters["oscillator_spin_interaction_coefficient"]
    γₗ = parameters["oscillator_damping_coefficient_left"]
    γᵣ = parameters["oscillator_damping_coefficient_right"]
    ω = parameters["oscillator_frequency"]
    T = parameters["temperature"]
    osc_dim = parameters["oscillator_space_dimension"]

    # - intervallo temporale delle simulazioni
    time_step = parameters["simulation_time_step"]

    # Costruzione della catena
    # ========================
    n_spin_sites = parameters["number_of_spin_sites"]

    # Leggo dal file temporaneo i dati prodotti dall'altro script.
    tmpfilename = replace(parameters["filename"], ".json" => ".dat.tmp")
    outfilename = replace(parameters["filename"], ".json" => ".dat")
    MPSdata = DataFrame(CSV.File(tmpfilename))

    time_step_list = MPSdata[:, :time]
    occ_n_MPS = hcat(MPSdata[:, :occ_n_left],
                     [MPSdata[:, Symbol("occ_n_spin$n")]
                      for n ∈ 1:n_spin_sites]...,
                     MPSdata[:, :occ_n_right])
    bond_dimensions = hcat([MPSdata[:, Symbol("bond_dim$n")]
                            for n ∈ 1:(n_spin_sites+1)]...)

    @info "Costruzione dell'equazione di Lindblad."
    # Calcolo la soluzione esatta, integrando l'equazione di Lindblad.
    bosc = FockBasis(osc_dim)
    bspin = SpinBasis(1//2)
    bcoll = tensor(bosc, repeat([bspin], n_spin_sites)..., bosc)

    function numop(i::Int)
      if i == 1 || i == n_spin_sites+2
        op = embed(bcoll, i, number(bosc))
      elseif i ∈ 1 .+ (1:n_spin_sites)
        op = embed(bcoll, i, QuantumOptics.projector(spinup(bspin)))
      else
        throw(DomainError)
      end
      return op
    end
    hspinloc(i) = embed(bcoll, 1+i, sigmaz(bspin))
    hspinint(i) = tensor(one(bosc),
                         repeat([one(bspin)], i-1)...,
                         tensor(sigmap(bspin), sigmam(bspin)) +
                           tensor(sigmam(bspin), sigmap(bspin)),
                         repeat([one(bspin)], n_spin_sites-i-1)...,
                         one(bosc))

    # Mi serve una matrice nulla per poter dare un elemento neutro
    # alle funzioni `sum(...)` qui sotto. Il modo più veloce per
    # calcolarla mi sembra il seguente.
    zeroop = 0. * hspinloc(1)

    # Gli operatori Hamiltoniani
    Hoscsx = ω * numop(1)
    Hintsx = κ * tensor(create(bosc)+destroy(bosc),
                        sigmax(bspin),
                        repeat([one(bspin)], n_spin_sites-1)...,
                        one(bosc))
    Hspin = 0.5ε*sum(hspinloc.(1:n_spin_sites); init=zeroop) -
            0.5*sum(hspinint.(1:n_spin_sites-1); init=zeroop)
    Hintdx = κ * tensor(one(bosc),
                        repeat([one(bspin)], n_spin_sites-1)...,
                        sigmax(bspin),
                        create(bosc)+destroy(bosc))
    Hoscdx = ω * numop(2+n_spin_sites)

    H = Hoscsx + Hintsx + Hspin + Hintdx + Hoscdx

    # La posizione di partenza del sistema è determinata da alcuni
    # parametri nel file JSON.
    name = lowercase(parameters["left_oscillator_initial_state"])
    if name == "thermal"
      if T == 0
        matL = QuantumOptics.projector(fockstate(bosc, 0))
      else
        matL = thermalstate(ω*number(bosc), T)
      end
    elseif name == "empty"
      matL = QuantumOptics.projector(fockstate(bosc, 0))
    elseif occursin(r"^fock", name)
      j = parse(Int, replace(name, "fock" => ""))
      matL = QuantumOptics.projector(fockstate(bosc, j))
    else
      throw(DomainError(statename,
                        "Stato non riconosciuto; scegliere tra «empty», «fockN» "*
                        "oppure «thermal»."))
    end
    name = lowercase(parameters["spin_initial_state"])
    if name == "up"
      firstspin = QuantumOptics.projector(spinup(bspin))
    elseif name == "down" || name == "dn" || name == "empty"
      firstspin = QuantumOptics.projector(spindown(bspin))
    elseif name == "x" || name == "sigmax"
      firstspin = sigmax(bspin)
    else
      throw(DomainError(state,
                        "Stato non riconosciuto; scegliere tra «up», "*
                        "«down», oppure «x»."))
    end
    ρ₀ = tensor(matL,
                firstspin,
                repeat([QuantumOptics.projector(spindown(bspin))],
                       n_spin_sites-1)...,
                QuantumOptics.projector(fockstate(bosc, 0)))

    if T == 0
      n = 0
    else
      n = (ℯ^(ω / T) - 1)^(-1)
    end
    rates = [γₗ * (n+1), γₗ * n, γᵣ]
    jump = [embed(bcoll, 1, destroy(bosc)),
            embed(bcoll, 1, create(bosc)),
            embed(bcoll, 2+n_spin_sites, destroy(bosc))]

    function fout(t, rho)
      occ_n = [real(QuantumOptics.expect(N, rho))
               for N in numop.(1:(2+n_spin_sites))]
      return [occ_n..., QuantumOptics.tr(rho)]
    end

    @info "Integrazione dell'equazione di Lindblad in corso."
    τ = time_step_list[begin+1] - time_step_list[begin]
    tout, output = timeevolution.master(time_step_list,
                                        ρ₀, H, jump;
                                        rates=rates,
                                        fout=fout,
                                        dt=0.1τ)
    output = permutedims(hcat(output...))
    occ_n_numeric = real.(output[:,1:end-1])
    norm_numeric = real.(output[:,end])

    @info "Scrittura dei risultati su file."
    dict = Dict(:time => time_step_list)
    for (col, name) ∈ enumerate([:occ_n_left_MPS;
                                 [Symbol("occ_n_spin$(n)_MPS")
                                  for n ∈ 1:n_spin_sites];
                                 :occ_n_right_MPS])
      push!(dict, name => occ_n_MPS[:,col])
    end

    for (col, name) ∈ enumerate([:occ_n_left_numeric;
                                 [Symbol("occ_n_spin$(n)_numeric")
                                  for n ∈ 1:n_spin_sites];
                                 :occ_n_right_numeric])
      push!(dict, name => occ_n_numeric[:,col])
    end

    diff_occ_n = occ_n_numeric .- occ_n_MPS
    for (col, name) ∈ enumerate([:diff_occ_n_left;
                                  [Symbol("diff_occ_n_spin$n")
                                   for n ∈ 1:n_spin_sites];
                                  :diff_occ_n_right])
      push!(dict, name => diff_occ_n[:,col])
    end

    for (col, name) ∈ enumerate([Symbol("bond_dim$n")
                                  for n ∈ 1:n_spin_sites+1])
      push!(dict, name => bond_dimensions[:,col])
    end

    norm_MPS = MPSdata[:, :full_trace]
    push!(dict, :trace_MPS => norm_MPS)
    push!(dict, :trace_numeric => norm_numeric)

    CSV.write(outfilename, DataFrame(dict))

    # Salvo i risultati nei grandi contenitori
    push!(timesteps_super, time_step_list)
    push!(occ_n_MPS_super, occ_n_MPS)
    push!(occ_n_numeric_super, occ_n_numeric)
    push!(diff_occ_n_super, diff_occ_n)
    push!(bond_dimensions_super, bond_dimensions)
    push!(normalisation_super, norm_MPS)
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
  for (occn, filename) ∈ zip([occ_n_numeric_super,
                              occ_n_MPS_super,
                              diff_occ_n_super],
                             ["populations_QuantumOptics",
                              "populations_ITensors",
                              "populations_ITensors_QuantumOptics_diff"])
    @pgf begin
      grp = GroupPlot({
                       group_opts...,
                       xlabel = L"\lambda t",
                       ylabel = L"\langle n_i(t)\rangle",
                      })
      for (t, data, p) ∈ zip(timesteps_super,
                             occn,
                             parameter_lists)
        ax = Axis({title = filenamett(p)})
        N = size(data, 2)
        for (y, c, ls) ∈ zip(eachcol(data),
                             readablecolours(N),
                             ["dashed"; repeat(["solid"], N-2); "dashed"])
          plot = Plot({ color = c, $ls }, Table([t, y]))
          push!(ax, plot)
        end
        push!(ax, Legend( ["L"; string.(1:N-2); "R"] ))
        push!(grp, ax)
      end
      pgfsave("$filename.pdf", grp)
    end
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
      N = size(data, 2) # whatever this is. I'm not checking
      for (y, c) ∈ zip(eachcol(data), readablecolours(N))
        plot = Plot({ color = c }, Table([t, y]))
        push!(ax, plot)
      end
      push!(ax, Legend( consecutivepairs(1:N) ))
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

  cd(prev_dir) # Il lavoro è completato: ritorna alla cartella iniziale.
  return
end
