from network import Network
from lossfunctions import MeanSquaredError
from metrics import BinaryAccuracy
from validation import kfold_cv
from itertools import product
import multiprocessing as mp
from tqdm import tqdm

def grid_search_cv(
    model_shape: Network,
    x,
    y,
    n_folds=3,
    metric=BinaryAccuracy(),
    loss=MeanSquaredError(),
    eta=[1e-3],
    alpha=[0.9],
    nesterov=[False, True],
    reg_type=[None],
    reg_val=[0],
    epochs=1000,
    verbose=False,
    scaler=None,
    workers=4,
):
    """
    Esegue la ricerca a griglia con validazione incrociata.

    Parametri:
    model_shape: Oggetto Network con l'architettura di rete desiderata.
    x: Dati di training.
    y: Etichette di training.
    n_folds: Numero di fold per la validazione incrociata.
    metric: Metrica per la valutazione.
    loss: Funzione di perdita da utilizzare.
    eta: Tasso di apprendimento.
    alpha: Forza del momento
    nesterov (booleano): Indica se usare il momento di nesterov o no.
    reg_type: Tipo di regolarizzazione.
    reg_val: Valore della regolarizzazione.
    epochs: Numero di epoche di training.
    verbose: Stampa la barra di progresso.
    scaler: Scaler per la normalizzazione dei dati.
    workers: Numero di worker per la parallelizzazione.

    Output:
    Dizionario con i risultati della ricerca a griglia.
    """

    # Determina le combinazioni di parametri da testare
    # Se non viene usata la regolarizzazione, combina solo parametro di apprendimento e impulso
    if reg_val == 0 or reg_type is None:
        parameters = product(eta, alpha, nesterov)
    else:
        parameters = product(eta, alpha, nesterov, reg_type, reg_val)
    
    # Inizializzazione delle liste per memorizzare le metriche
    manager = mp.Manager()
    return_dict = manager.dict()

    print("Gridsearch: esplorando " + str(len(list(parameters))) + " combinazioni.")
    parameters = list(parameters)  # Ricrea la lista dopo averne calcolato la lunghezza
    bar = tqdm(total=len(parameters))

    # Funzione per eseguire un singolo processo di validazione incrociata
    def run_kfold_cv(params):
        model = Network(model_shape.layers[0].units, params.get("reg_type", None))
        for layer in model_shape.layers[1:]:
            model.add_layer(layer.units, layer.activation)
        kfold_cv(
            model, x, y, k=n_folds, eta=params["eta"],
            alpha=params["alpha"], nesterov=params["nesterov"], epochs=epochs,
            metric=metric, loss=loss, scaler=scaler, verbose=verbose,
            return_dict=return_dict, pid=params["id"]
        )

    # Avvia i processi di validazione incrociata in parallelo
    with mp.Pool(workers) as pool:
        for count, par in enumerate(parameters):
            params = {
                "eta": par[0],
                "alpha": par[1],
                "nesterov": par[2],
                "reg_type": par[3] if len(par) > 3 else None,
                "reg_val": par[4] if len(par) > 4 else 0,
                "id": f"proc-{count}"
            }
            pool.apply_async(run_kfold_cv, args=(params,), callback=lambda _: bar.update(1))

    bar.close()

    # Unisce i risultati delle configurazioni di parametri
    merged = {key: (params[key], return_dict[key]) for key in return_dict.keys()}

    return merged

def get_reg_as_string(reg_type):
    return "None" if reg_type is None else "L1" if "L1" in str(reg_type) else "L2"