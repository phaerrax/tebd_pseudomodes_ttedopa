#!/usr/bin/julia

using ITensors
using LaTeXStrings
using Base.Filesystem
using DataFrames
using CSV
using Plots
using QuadGK

# Se lo script viene eseguito su Qtech, devo disabilitare l'output
# grafico altrimenti il programma si schianta.
if gethostname() == "qtech.fisica.unimi.it" ||
  gethostname() == "qtech2.fisica.unimi.it"
  ENV["GKSwstype"] = "100"
  @info "Esecuzione su server remoto. Output grafico disattivato."
else
  delete!(ENV, "GKSwstype")
  # Se la chiave "GKSwstype" non esiste non succede niente.
end

rootdirname = "simulazioni_tesi"
sourcepath = Base.source_path()
# Cartella base: determina il percorso assoluto del file in esecuzione, e
# rimuovi tutto ciò che segue rootdirname.
ind = findfirst(rootdirname, sourcepath)
rootpath = sourcepath[begin:ind[end]]
# `rootpath` è la cartella principale del progetto.
libpath = joinpath(rootpath, "lib")

let  
  Ω = 10
  κ = 1

  pseudomodesJ(ω,γ,T) = κ^2 * γ/(2π) * (0.5*(coth(0.5Ω/T) + 1) / hypot(0.5γ, ω-Ω)^2 +
                                       0.5*(coth(0.5Ω/T) - 1) / hypot(0.5γ, ω+Ω)^2)
  ttedopaJ(ω,γ,T) = κ^2 * γ/(2π) * 0.5*(coth(0.5ω/T) + 1) * (hypot(0.5γ, ω-Ω)^(-2) -
                                                            hypot(0.5γ, ω+Ω)^(-2))

  Ts = 1e-4:4e-4:4
  γs = 0.1:0.2:1
  overlap = [[first(quadgk(ω -> abs(pseudomodesJ(ω,γ,T) - ttedopaJ(ω,γ,T)), -Inf, 0, Inf))
              for T ∈ Ts] for γ ∈ γs]

  plt = plot(Ts, overlap[1], label="γ=$(γs[1])")
  for (γ,o) ∈ zip(γs[2:end], overlap[2:end])
    plot!(plt, Ts, o, label="γ=$γ")
  end
  display(plt)

  print([(γ, first(quadgk(ω -> abs(pseudomodesJ(ω,γ,20) - ttedopaJ(ω,γ,20)), -Inf, 0, Inf))) for γ ∈ 0.1:0.1:1])
  return
end
