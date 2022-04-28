#!/usr/bin/julia

using JSON
using Base.Filesystem

root_path = dirname(dirname(Base.source_path()))
lib_path = root_path * "/lib"
# Sali di due cartelle. root_path è la cartella principale del progetto.
include(lib_path * "/utils.jl")
include(lib_path * "/plotting.jl")
include(lib_path * "/spin_chain_space.jl")
include(lib_path * "/harmonic_oscillator_space.jl")
include(lib_path * "/operators.jl")

let
  parameter_lists = load_parameters(ARGS)
  prev_dir = pwd()
  if isdir(ARGS[1])
    cd(ARGS[1])
  end

  for parameters in parameter_lists
    newpar = parameters

    κ = pop!(newpar, "oscillator_spin_interaction_coefficient")
    γ = pop!(newpar, "oscillator_damping_coefficient")
    Ω = pop!(newpar, "oscillator_frequency")
    T = pop!(newpar, "temperature")
    ν(ω,T) = T == 0 ? 0.0 : (ℯ^(ω/T)-1)^(-1)

    # Transform the pseudomode into two zero-temperature new modes.
    push!(newpar, "oscillatorL1_frequency" => Ω)
    push!(newpar, "oscillatorL1_spin_interaction_coefficient" => κ*sqrt(1+ν(T,Ω)) )
    push!(newpar, "oscillatorL1_damping_coefficient" => γ)
    push!(newpar, "oscillatorL2_frequency" => -Ω)
    push!(newpar, "oscillatorL2_spin_interaction_coefficient" => κ*sqrt(ν(T,Ω)) )
    push!(newpar, "oscillatorL2_damping_coefficient" => γ)
    push!(newpar, "oscillatorR_frequency" => Ω)
    push!(newpar, "oscillatorR_spin_interaction_coefficient" => κ)
    push!(newpar, "oscillatorR_damping_coefficient" => γ)
    push!(newpar, "temperature" => 0)
    push!(newpar, "oscillators_interaction_coefficient" => 0)

    # Set new initial oscillator states as empty.
    pop!(newpar, "left_oscillator_initial_state")
    push!(newpar, "left_oscillator_initial_state" => "empty")

    # Save old parameters for quick reference.
    push!(newpar, "original_oscillator_spin_interaction_coefficient" => κ)
    push!(newpar, "original_oscillator_damping_coefficient" => γ)
    push!(newpar, "original_oscillator_frequency" => Ω)
    push!(newpar, "original_temperature" => T)

    # Remove the filename from the JSON file.
    newfile = pop!(newpar, "filename")

    open(newfile, "w") do f
      JSON.print(f, newpar, 2)
    end
  end
  cd(prev_dir)
end
