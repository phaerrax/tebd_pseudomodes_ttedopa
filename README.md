# tebd_pseudomodes_ttedopa
Il codice sorgente delle simulazioni svolte per la tesi magistrale.

## Esecuzione
Gli script richiedono dei file JSON, contenenti i parametri fisici e tecnici per la simulazione, cioè del formato

	{
		"chiave1": valore1,
		"chiave2": valore2,
		...
		"chiaveN": valoreN
	}

In questo modo non è necessario modificare il codice del programma solo per provare nuovi valori dei parametri.
Questi file possono essere forniti a ciascun programma tramite la riga di comando in due modi:
1) semplicemente scrivendoli uno dopo l'altro, ad esempio: `./pseudomodes.jl parametri1.json parametri2.json parametri3.json`;
2) fornendo una cartella, all'interno della quale verranno letti *tutti* i file che terminano per `.json`.

Il programma produce come output una tabella CSV con i dati ricavati dalla simulazione (un file per ciascun file di parametri fornito), e alcuni grafici in formato PNG: nel caso 1), i file sono salvati nella cartella da dove è stato lanciato il programma, nel caso 2) invece i file si troveranno nella cartella che contiene i file JSON e che è stata fornita come argomento alla linea di comando.

Tutte le grandezze fisiche sono da intendersi come quantità ridotte (come spiegato nella tesi), vale a dire che il coefficiente λ dell'interazione tra gli spin vale sempre 1. Gli script richiedono determinati parametri, che devono essere forniti nei file JSON.
Per ciascuno script è presente un file, il cui nome termina in `_sample.json`, in cui sono elencati tutti i parametri necessari per eseguire la simulazione.

Nella cartella `tests` si trovano inoltre alcuni script usati per testare alcune particolarità dei sistemi o per fini diagnostici.

### Simulazioni di base
I seguenti script eseguono la simulazione di una catena di spin, chiusa o isolata (quindi senza ambienti esterni strutturati).

##### `closed_spin_chain/closed_spin_chain.jl`
Questo programma simula una catena di spin isolata. La cartella contiene anche `closed_spin_chain_exact.jl` che calcola la dinamica esatta del sistema.

##### `open_spin_chain_markov/open_spin_chain_markov.jl`, `open_spin_chain_markov_alt/open_spin_chain_markov_alt.jl`
Questi due script simulano una catena di spin i cui estremi interagiscono con bagni termici seguendo una dinamica markoviana; la differenza tra i due sistemi consiste negli operatori di Lindblad che descrivono la dissipazione.

### Simulazioni con pseudomodi
I seguenti programmi eseguono la simulazione con gli ambienti rappresentati da uno o due pseudomodi.

##### `pseudomodes/pseudomodes.jl`
Questo programma simula la catena di spin la cui interazione con i bagni termici è modellata tramite degli oscillatori armonici, smorzati secondo una dinamica markoviana, collegati agli estremi della catena.

##### `pseudomodes2/pseudomodes2.jl`
Questo programma è come il precedente, ma l'ambiente di sinistra è descritto tramite due pseudomodi anziché uno solo (quello di destra rimane un solo pseudomodo). Al momento è necessario impostare il coefficiente di interazione tra i due pseudomodi a zero: non è detto che i risultati abbiano senso se esso non è nullo.

### Simulazioni con T-TEDOPA
Gli script in `ttedopa` eseguono la simulazione della solita catena di spin collegata a due bagni termici, che vengono trasformati in una catena (teoricamente infinita) di oscillatori armonici secondo l'algoritmo T-TEDOPA.

##### `ttedopa/generic.jl`
In questo script i due bagni possono essere specificati in due modi esclusivi (all'interno del file JSON):

* puntando a un altro file di testo che contiene due colonne di nome `loc` e `int`:
	- colonna `loc` = coefficienti locali degli oscillatori,
	- colonna `int` = coefficienti di interazione tra gli oscillatori (il primo dei quali è il coefficiente di interazione spin-oscillatore);
* tramite un sotto-dizionario in cui la densità spettrale è indicata nella forma di una stringa (che dev'essere una valida funzione di Julia, nella variabile `x`, con una lista `a` di parametri) unita agli altri parametri come ad esempio la temperatura e la frequenza massima del supporto della funzione.

Un file JSON non può contenere entrambe le descrizioni per un solo ambiente, ma i due ambienti in un sistema possono essere descritti in modi diversi.

Il file `ttedopa/test_generic/generic_sample.json` illustra entrambe le modalità. Inoltre, `ttedopa/gen_coefficients.jl` genera un file di coefficienti (nel giusto formato per `generic.jl`) a partire da una densità descritta in questo modo.

##### `ttedopa/lorentzian.jl`, `ttedopa/double_lorentzian.jl`
Questi script sono specializzati al caso in cui la densità spettrale sia composta, rispettivamente, da una Lorentziana antisimmetrizzata, o dalla combinazione lineare di due tali funzioni.