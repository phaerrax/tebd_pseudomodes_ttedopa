using ITensors

# Hamiltoniani locali
function Hlocal(s::Index)
  if SiteType("S=1/2") ∈ sitetypes(s)
    h = 0.5op("σz", s)
  elseif SiteType("Osc") ∈ sitetypes(s)
    h = op("N", s)
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return h
end

# Hamiltoniani di interazione
function Hinteraction(s1::Index, s2::Index)
  if SiteType("S=1/2") ∈ sitetypes(s1) && SiteType("S=1/2") ∈ sitetypes(s2)
    h = -0.5 * (op("σ-", s1) * op("σ+", s2) +
                op("σ+", s1) * op("σ-", s2))
  elseif SiteType("Osc") ∈ sitetypes(s1) && SiteType("S=1/2") ∈ sitetypes(s2)
    h = op("X", s1) * op("σx", s2)
  elseif SiteType("S=1/2") ∈ sitetypes(s1) && SiteType("Osc") ∈ sitetypes(s2)
    h = op("σx", s1) * op("X", s2)
  elseif SiteType("Osc") ∈ sitetypes(s1) && SiteType("Osc") ∈ sitetypes(s2)
    h = (op("a+", s1) * op("a-", s2) +
         op("a-", s1) * op("a+", s2))
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return h
end

# Lindblad locali
function ℓlocal(s::Index)
  if SiteType("vecS=1/2") ∈ sitetypes(s)
    ℓ = 0.5im * (op("σz:Id", s) - op("Id:σz", s))
  elseif SiteType("vecOsc") ∈ sitetypes(s)
    ℓ = im * (op("N:Id", s) - op("Id:N", s))
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return ℓ
end

# Lindblad di interazione
function ℓinteraction(s1::Index, s2::Index)
  if SiteType("vecS=1/2") ∈ sitetypes(s1) &&
     SiteType("vecS=1/2") ∈ sitetypes(s2)
    ℓ = -0.5im * (op("σ-:Id", s1) * op("σ+:Id", s2) +
                  op("σ+:Id", s1) * op("σ-:Id", s2) -
                  op("Id:σ-", s1) * op("Id:σ+", s2) -
                  op("Id:σ+", s1) * op("Id:σ-", s2))
  elseif SiteType("vecOsc") ∈ sitetypes(s1) &&
         SiteType("vecS=1/2") ∈ sitetypes(s2)
    ℓ = im * (op("asum:Id", s1) * op("σx:Id", s2) -
              op("Id:asum", s1) * op("Id:σx", s2))
  elseif SiteType("vecS=1/2") ∈ sitetypes(s1) &&
         SiteType("vecOsc") ∈ sitetypes(s2)
    ℓ = im * (op("σx:Id", s1) * op("asum:Id", s2) -
              op("Id:σx", s1) * op("Id:asum", s2))
  else
    throw(DomainError((s1, s2), "SiteType non riconosciuti."))
  end
  return ℓ
end
