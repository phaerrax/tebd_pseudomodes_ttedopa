#!/usr/bin/julia

using ITensors, DataFrames, LaTeXStrings, CSV, Plots, QuadGK
using PseudomodesTTEDOPA

disablegrifqtech()

let  
  Ω = 10
  κ = 0.5

  pseudomodesJ(ω,γ,T) = κ^2 * γ/(2π) * (0.5*(coth(0.5Ω/T) + 1) / hypot(0.5γ, ω-Ω)^2 +
                                       0.5*(coth(0.5Ω/T) - 1) / hypot(0.5γ, ω+Ω)^2)
  ttedopaJ(ω,γ,T) = κ^2 * γ/(2π) * 0.5*(coth(0.5ω/T) + 1) * (hypot(0.5γ, ω-Ω)^(-2) -
                                                            hypot(0.5γ, ω+Ω)^(-2))

  Ts = 0e-4:1e-2:4
  γs = 0:0.2:1
  overlap = [[first(quadgk(ω -> abs(pseudomodesJ(ω,γ,T) - ttedopaJ(ω,γ,T)), -Inf, 0, Inf))
              for T ∈ Ts] for γ ∈ γs]

  plt = plot(Ts, overlap[1], label="γ=$(γs[1])")
  for (γ,o) ∈ zip(γs[2:end], overlap[2:end])
    plot!(plt, Ts, o, label="γ=$γ")
  end
  table = Dict(:temperature => collect(Ts))
  for (γ,o) ∈ zip(γs, overlap)
    push!(table, Symbol(string("gamma", γ)) => o)
  end
  filename = "overlap_T_gamma.dat"
  CSV.write(filename, DataFrame(table))
  display(plt)

  print([(γ, first(quadgk(ω -> abs(pseudomodesJ(ω,γ,20) - ttedopaJ(ω,γ,20)), -Inf, 0, Inf))) for γ ∈ 0.1:0.1:1])
  return
end
