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
1) semplicemente scrivendoli uno dopo l'altro, ad esempio: `./chain_with_damped_oscillators.jl parametri1.json parametri2.json parametri3.json`;
2) fornendo una cartella, all'interno della quale verranno letti *tutti* i file .json.
Il programma produce come output una tabella CSV con tutti i dati ricavati dalla simulazione (un file per ciascun file di parametri fornito), e alcuni grafici in formato PNG: nel caso 1), i file sono salvati nella cartella da dove è stato lanciato il programma, nel caso 2) invece i file si troveranno nella cartella che contiene i file JSON e che è stata fornita come argomento alla linea di comando.

#### Parametri richiesti
Tutte le grandezze fisiche sono da intendersi come quantità ridotte (come spiegato nella tesi), vale a dire che il coefficiente λ dell'interazione tra gli spin vale sempre 1. Per tutti gli script sono richiesti i parametri:
- `spin_excitation_energy`
- `number_of_spin_sites`
- `simulation_end_time`
- `simulation_time_step`
- `skip_steps`
- `chain_initial_state`
- `MP_maximum_bond_dimension`
- `MP_compression_error`
- `TS_expansion_order`

In aggiunta, `damped_chain.jl` richiede anche
- `spin_damping_coefficient`
- `temperature`

`chain_with_damper_oscillators.jl` richiede
- `oscillator_spin_interaction_coefficient`
- `oscillator_damping_coefficient_left`
- `oscillator_damping_coefficient_right`
- `oscillator_frequency`
- `temperature`

e infine `unitary_model` richiede
- `number_of_oscillators_left`
- `number_of_oscillators_right`
- `frequency_cutoff`
- `spectral_density_peak`
- `spectral_density_half_width`
- `temperature`
- `PolyChaos_nquad`

Per ogni script è presente un file, il cui nome termina in `_sample.json`, con un elenco di parametri di esempio.
