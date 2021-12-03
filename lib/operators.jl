using ITensors

# Hamiltoniani locali
function Hlocal(s::Index)
  if SiteType("S=1/2") ∈ sitetypes(s)
    h = op("σz", s)
  elseif SiteType("Osc") ∈ sitetypes(s)
    h = op("N", s)
  elseif SiteType("vecS=1/2") ∈ sitetypes(s)
    h = op("σz:Id", s) - op("Id:σz", s)
  elseif SiteType("vecOsc") ∈ sitetypes(s)
    h = op("N:Id", s) - op("Id:N", s)
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
function Llocal(s::Index)
  if SiteType("vecS=1/2") ∈ sitetypes(s)
    L = im * (op("σz:Id", s) - op("Id:σz", s))
  elseif SiteType("vecOsc") ∈ sitetypes(s)
    L = im * (op("N:Id", s) - op("Id:N", s))
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return L
end

# Lindblad di interazione
function Linteraction(s1::Index, s2::Index)
  if SiteType("vecS=1/2") ∈ sitetypes(s1) && SiteType("vecS=1/2") ∈ sitetypes(s2)
    L = -0.5im * (op("σ-:Id", s1) * op("σ+:Id", s2) +
                  op("σ+:Id", s1) * op("σ-:Id", s2) -
                  op("Id:σ-", s1) * op("Id:σ+", s2) -
                  op("Id:σ+", s1) * op("Id:σ-", s2))
  elseif SiteType("vecOsc") ∈ sitetypes(s1) && SiteType("vecS=1/2") ∈ sitetypes(s2)
  elseif SiteType("vecS=1/2") ∈ sitetypes(s1) && SiteType("vecOsc") ∈ sitetypes(s2)
  else
    throw(DomainError(s, "SiteType non riconosciuto."))
  end
  return L
end
