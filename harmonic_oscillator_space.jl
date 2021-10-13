using ITensors
using LinearAlgebra

# Spazio degli oscillatori
# ------------------------
osc_dim = 8
ITensors.space(::SiteType"vecOsc") = osc_dim^2

# Operatori di scala e affini
a⁻ = diagm(1 => [sqrt(j) for j = 1:osc_dim-1])
a⁺ = diagm(-1 => [sqrt(j) for j = 1:osc_dim-1])
a⁺[end,end] = 1
num = diagm(0 => 0:osc_dim-1)
Iₒ = Matrix{Int}(I, osc_dim, osc_dim)

function ITensors.state(::StateName"Emp:Emp", ::SiteType"vecOsc")
  empty = [i == 1 ? 1 : 0 for i = 1:osc_dim]
  return kron(empty, empty)
end
# Lo stato di equilibrio termico deve essere definito nello script principale,
# perché serve conoscere T ed ω
ITensors.state(::StateName"veca+", ::SiteType"vecOsc") = vcat(a⁺[:])
ITensors.state(::StateName"veca-", ::SiteType"vecOsc") = vcat(a⁻[:])
ITensors.state(::StateName"vecnum", ::SiteType"vecOsc") = vcat(num[:])
ITensors.state(::StateName"vecid", ::SiteType"vecOsc") = vcat(Iₒ[:])

# Operatori semplici sullo spazio degli oscillatori
ITensors.op(::OpName"id:id", ::SiteType"vecOsc") = kron(Iₒ, Iₒ)
# - interazione con la catena
ITensors.op(::OpName"id:asum", ::SiteType"vecOsc") = kron(Iₒ, a⁺ + a⁻)
ITensors.op(::OpName"asum:id", ::SiteType"vecOsc") = kron(a⁺ + a⁻, Iₒ)
# - Hamiltoniano del sistema libero
ITensors.op(::OpName"num:id", ::SiteType"vecOsc") = kron(num, Iₒ)
ITensors.op(::OpName"id:num", ::SiteType"vecOsc") = kron(Iₒ, num)
# - termini di dissipazione
ITensors.op(::OpName"a+T:a-", ::SiteType"vecOsc") = kron(transpose(a⁺), a⁻)
ITensors.op(::OpName"a-T:a+", ::SiteType"vecOsc") = kron(transpose(a⁻), a⁺)

# Composizione dell'Hamiltoniano per i termini degli oscillatori
# - termini locali dell'Hamiltoniano
function ITensors.op(::OpName"H1loc", ::SiteType"vecOsc", s::Index)
  h = op("num:id", s) - op("id:num", s)
  return im * h
end
# - termini di smorzamento
function ITensors.op(::OpName"damping", ::SiteType"vecOsc", s::Index; ω::Number, T::Number)
  if T == 0
    n = 0
  else
    n = (ℯ^(ω / T) - 1)^(-1)
  end
  d = (n + 1) * (op("a+T:a-", s) - 0.5 * (op("num:id", s) + op("id:num", s))) +
      n * (op("a-T:a+", s) - 0.5 * (op("num:id", s) + op("id:num", s)) - op("id:id", s))
  return d
end
