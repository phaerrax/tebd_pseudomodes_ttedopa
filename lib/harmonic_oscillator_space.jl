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

function ITensors.state(::StateName"canon", ::SiteType"vecOsc"; n::Int)
  v = [i == n ? 1 : 0 for i=1:osc_dim]
  return kron(v, v)
end

ITensors.state(::StateName"Emp:Emp", st::SiteType"vecOsc") = state(StateName("canon"), st; n=1)
ITensors.state(::StateName"veca+", ::SiteType"vecOsc") = vcat(a⁺[:])
ITensors.state(::StateName"veca-", ::SiteType"vecOsc") = vcat(a⁻[:])
ITensors.state(::StateName"vecnum", ::SiteType"vecOsc") = vcat(num[:])
ITensors.state(::StateName"vecid", ::SiteType"vecOsc") = vcat(Iₒ[:])
function ITensors.state(::StateName"ThermEq", ::SiteType"vecOsc", s::Index; ω, T)
  if T == 0
    mat = kron([1; repeat([0], osc_dim-1)],
               [1; repeat([0], osc_dim-1)])
  else
    mat = exp(-ω / T * num)
    mat /= tr(mat)
  end
  return vcat(mat[:])
end

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

# - proiezione sugli autostati dell'operatore numero
# Il sito è uno solo quindi basta usare i vettori della base canonica
function osc_levels_proj(site::Index{Int64}, level::Int)
  return MPS([ITensor(state(StateName("canon"),
                            SiteType("vecOsc");
                            n=level),
                      site)])
end

function ITensors.state(::StateName"mat_comp", ::SiteType"vecOsc", s::Index; j::Int, k::Int)
  return kron([i == j ? 1 : 0 for i=1:osc_dim],
              [i == k ? 1 : 0 for i=1:osc_dim])
end
