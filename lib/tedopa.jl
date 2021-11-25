using PolyChaos
using QuadGK

function thermalisedJ(J::Function,
                      ω::Real,
                      T::Real,
                      support::Tuple{Real, Real})
  # Questa funzione restituisce la densità spettrale termalizzata, a partire da
  # una J(ω) già antisimmetrica attorno a ω=0 (se non lo è, è necessario farlo
  # prima di chiamare questa funzione).
  #
  # Calcolo il "fattore termico" con cui moltiplicare J; se T=0, per evitare
  # divisioni per zero, scrivo a parte il fattore: lim T->0 coth(x/T)=Θ(x).
  # Si potrebbe anche scrivere f=1 e basta, assumendo che chi chiama la funzione
  # abbia usato un `support` che escluda già (-∞,0), ma così almeno sto sicuro. 
  if T == 0
    f = 0.5(1 + sign(ω))
  else
    f = 0.5(1 + coth(0.5 * ω / T))
  end
  return f * J(ω)
end

function chainmap_coefficients(J::Function,
                               support::Tuple{Real, Real},
                               L::Int;
                               Nquad::Int)
  #=
  Calcola i coefficienti dell'Hamiltoniana ottenuta dalla "chain map", a
  partire da una data funzione spettrale.

  Argomenti
  ---------
  · `J::Function`: una funzione dall'intervallo specificato in `support` a 
    valori non negativi.

  · `support::Tuple{Real, Real}`: una coppia di numeri reali a indicare gli
    estremi, in ordine, dell'intervallo in cui è definita la funzione J.

  · `L::Int`: la lunghezza della serie di oscillatori che si desidera
    costruire, nel senso del numero di oscillatori presenti.

  · `Nquad::Int`: un parametro da passare ad OrthoPoly, che specifica il numero
    di nodi usati nel metodo delle quadrature per calcolare gli integrali. Per
    modificare la precisione dei risultati, modificare questo parametro.
  =#
  # Calcolo il coefficiente per l'operatore di interazione spin-oscillatore.
  if support[1] < 0 && support[2] > 0
    η = first(quadgk(J, support[1], 0, support[2]))
    # La documentazione di quadgk suggerisce di usare questa forma nel caso
    # in cui l'integranda sia singolare in qualche punto (in questo caso in 0).
  else
    η = first(quadgk(J, support[1], support[2]))
  end
  #=
  I valori aₙ e bₙ sono i coefficienti nella formula di ricorsione
  x πₙ(x) = πₙ₊₁(x) + aₙ πₙ(x) + bₙ πₙ₋₁(x)
  dove {πₙ}ₙ, con n ∈ ℕ è la sequenza di polinomi ortogonali monici
  associata alla funzione J; π₋₁ è il polinomio nullo.
  
  Per costruire una serie di L oscillatori, servono i coefficienti αᵢ e βᵢ da
  i=0 a i=L-1: siccome però gli array sono indicizzati partendo da 1 in Julia,
  mi serviranno α[1:L] e β[1:L].
  A partire da questi coefficienti, indicizzando i siti degli oscillatori da 1
  a L, le frequenze Ωᵢ di ciascun oscillatore e la costante di interazione κᵢ
  tra gli oscillatori (i,i+1) sono date da
  Ωᵢ = αᵢ            per i∈{1,…,L},
  κᵢ = sqrt(βᵢ₊₁)    per i∈{1,…,L-1}.
  =#
  measure = PolyChaos.Measure("measure", J, support, false, Dict())
  poly = PolyChaos.OrthoPoly("poly", L, measure; Nquad=Nquad)
  a = coeffs(poly)[:,1]
  b = coeffs(poly)[:,2]
  return a[1:L], sqrt.(b[2:L]), η
end
