using LinearAlgebra
using ITensors

# Spazio degli spin
# -----------------
ITensors.space(::SiteType"vecS=1/2") = 4

# Questi "stati" contengono sia degli stati veri e propri, come Up:Up e Dn:Dn,
# sia la vettorizzazione di alcuni operatori che mi servono per calcolare le
# osservabili sugli MPS (siccome sono vettori, devo definirli per forza così)
ITensors.state(::StateName"Up:Up", ::SiteType"vecS=1/2") = [ 1 0 0 0 ]
ITensors.state(::StateName"Dn:Dn", ::SiteType"vecS=1/2") = [ 0 0 0 1 ]
ITensors.state(::StateName"vecσx", ::SiteType"vecS=1/2") = [ 0 1 1 0 ]
ITensors.state(::StateName"vecσy", ::SiteType"vecS=1/2") = [ 0 im -im 0 ]
ITensors.state(::StateName"vecσz", ::SiteType"vecS=1/2") = [ 1 0 0 -1 ]
ITensors.state(::StateName"vecid", ::SiteType"vecS=1/2") = [ 1 0 0 1 ]

# Matrici di Pauli e affini
σˣ = [0 1
      1 0]
σʸ = [0  -im
      im 0  ]
σᶻ = [1 0
      0 -1]
σ⁺ = [0 1
      0 0]
σ⁻ = [0 0
      1 0]
I₂ = [1 0
      0 1]

# Operatori semplici sullo spazio degli spin
ITensors.op(::OpName"id:id", ::SiteType"vecS=1/2") = kron(I₂, I₂)

ITensors.op(::OpName"σz:id", ::SiteType"vecS=1/2") = kron(σᶻ, I₂)
ITensors.op(::OpName"id:σz", ::SiteType"vecS=1/2") = kron(I₂, σᶻ)

ITensors.op(::OpName"id:σ+", ::SiteType"vecS=1/2") = kron(I₂, σ⁺)
ITensors.op(::OpName"σ+:id", ::SiteType"vecS=1/2") = kron(σ⁺, I₂)
ITensors.op(::OpName"id:σ-", ::SiteType"vecS=1/2") = kron(I₂, σ⁻)
ITensors.op(::OpName"σ-:id", ::SiteType"vecS=1/2") = kron(σ⁻, I₂)

ITensors.op(::OpName"σx:σx", ::SiteType"vecS=1/2") = kron(σˣ,σˣ)
