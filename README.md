# simulazioni_tesi
Il codice sorgente delle simulazioni svolte per la tesi magistrale.

## Esecuzione
Gli script richiedono dei file JSON, contenenti i parametri fisici e tecnici per la simulazione, formattato come un dizionario Python, cioè

	{
		"chiave1": valore1,
		"chiave2": valore2,
		...
		"chiaveN": valoreN
	}
In questo modo non è necessario modificare il codice del programma solo per provare nuovi valori dei parametri.
Questi file possono essere forniti a ciascun programma tramite la riga di comando in due modi:
1) semplicemente scrivendoli uno dopo l'altro, ad esempio: `./lindblad.jl parametri1.json parametri2.json parametri3.json`;
2) fornendo una cartella, all'interno della quale verranno letti *tutti* i file .json.
Il programma produce come output una tabella CSV con tutti i dati ricavati dalla simulazione (un file per ciascun file di parametri fornito), e alcuni grafici in formato PNG: nel caso 1), i file sono salvati nella cartella da dove è stato lanciato il programma, nel caso 2) invece i file si troveranno nella cartella che contiene i file JSON e che è stata fornita come argomento alla linea di comando.

#### Gli script
Tutte le grandezze fisiche sono da intendersi come quantità ridotte (come spiegato nella tesi), vale a dire che il coefficiente λ dell'interazione tra gli spin vale sempre 1. Ciascuno script richiede determinati parametri nei file JSON forniti.
Nella cartella `tests` si trovano inoltre alcuni script usati per testare alcune particolarità dei sistemi o per fini diagnostici.

##### closed_spin_chain
Questo programma simula una catena di spin isolata. La cartella contiene anche `closed_spin_chain_exact.jl` che calcola la dinamica esatta del sistema.
I parametri richiesti sono
- `spin_excitation_energy`
- `number_of_spin_sites`
- `simulation_end_time`
- `simulation_time_step`
- `skip_steps`
- `chain_initial_state`
- `MP_maximum_bond_dimension`
- `MP_compression_error`
- `TS_expansion_order`

##### open_spin_chain_markov e open_spin_chain_markov_alt
Questi due script simulano una catena di spin i cui estremi interagiscono con bagni termici seguendo una dinamica markoviana; la differenza tra i due sistemi consiste negli operatori di Lindblad che descrivono la dissipazione.
I parametri richiesti sono
- `spin_damping_coefficient`
- `temperature`

##### pseudomodes
Questo programma simula la catena di spin la cui interazione con i bagni termici è modellata tramite degli oscillatori armonici, smorzati secondo una dinamica markoviana, collegati agli estremi della catena.
Richiede i parametri
- `left_oscillator_initial_state`
- `oscillator_space_dimension`
- `oscillator_spin_interaction_coefficient`
- `oscillator_damping_coefficient_left`
- `oscillator_damping_coefficient_right`
- `oscillator_frequency`
- `temperature`

##### ttedopa
Questo programma simula infine la catena di spin interagente con due bagni termici, il cui sistema è stato trasformato in una catena (teoricamente infinita) di oscillatori armonici secondo l'algoritmo T-TEDOPA.
- `left_oscillator_initial_state`
- `maximum_oscillator_space_dimension`
- `oscillator_space_dimensions_decay`
- `number_of_oscillators_left`
- `number_of_oscillators_right`
- `frequency_cutoff`
- `spectral_density_peak`
- `spectral_density_half_width`
- `spectral_density_overall_factor`
- `temperature`
- `PolyChaos_nquad`

Per ogni script è presente un file, il cui nome termina in `_sample.json`, con un elenco di parametri di esempio.
