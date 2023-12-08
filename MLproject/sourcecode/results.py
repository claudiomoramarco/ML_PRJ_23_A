import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, load_digits

# Funzione per la codifica one-hot
def one_hot_encoder(data, num_classes=10):
    """Converte un array di interi in una matrice diagonale one-hot."""
    return np.eye(num_classes)[data]

# Classe Error con metodi per valutazione errore
class Error:
    """Classe base per l'errore."""
    def validate(self, y_true, y_pred):
        """Calcola l'errore come differenza tra il valore previsto e quello atteso."""
        raise NotImplementedError("Must implement a subclass")

class MultiClassError(Error):
    """Errore per i problemi di classificazione multiclasse (i MONK)."""
    def validate(self, y_true, y_pred):
        y_true = np.argmax(y_true)
        y_pred = np.argmax(y_pred)
        if y_true == y_pred:
            return 0
        else:
            return 1

class SingleClassError(Error):
    """Errore per i problemi di classificazione binaria (i moon)."""
    def validate(self, y_true, y_pred):
        y_pred = np.round(y_pred).astype(int)
        return np.mean(y_true != y_pred)


# Caricamento dei MONK

def load_monk(dataset_number, test_size=0.2, validation=False):
    """Carica e preprocessa il MONK specificato."""
    file_base_path = r"C:\Users\anton\OneDrive\Desktop\X27 - UNIVERSITA'\43 - MACHINE LEARNING\MLproject\data\MONK\monks-{dataset_number}"
    train = pd.read_csv(f"{file_base_path}.train", sep=" ").drop(["a8"], axis=1)
    test = pd.read_csv(f"{file_base_path}.test", sep=" ").drop(["a8"], axis=1)

    enc = OneHotEncoder(handle_unknown="ignore")
    X_train = enc.fit_transform(train.drop("a1", axis=1)).toarray()
    y_train = train["a1"].values.reshape(-1, 1)

    X_test = enc.transform(test.drop("a1", axis=1)).toarray()
    y_test = test["a1"].values.reshape(-1, 1)

    return _split_data(X_train, y_train, X_test, y_test, test_size, validation)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

def load_moons(n_samples=1000, test_size=0.1, random_state=42, noise=0.2, validation=True):
    """
    Carica e prepara il dataset di prova 'moons'.

    Argomenti:
    n_samples (int): Numero di campioni da generare nel dataset. Default: 1000.
    test_size (float): Proporzione del dataset da utilizzare come set di test. Default: 0.1 (10%).
    random_state (int): Seme per la generazione casuale, per risultati riproducibili. Default: 42.
    noise (float): Quantità di rumore da aggiungere ai dati. Default: 0.2.
    validation (bool): Se dividere ulteriormente i dati in un set di validazione. Default: True.

    Output:
    x_train, x_val, x_test (ndarray): Set di dati di input per training, validazione e test.
    y_train, y_val, y_test (ndarray): Set di etichette corrispondenti per training, validazione e test.
    """

    # Genera il dataset 'moons' usando la funzione make_moons di scikit-learn.
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # Espande le dimensioni di x e y per soddisfare i requisiti di input di alcuni modelli.
    # In particolare x diventa tridimensionale e y bidimensionale.
    x = np.expand_dims(X, 2)
    y = np.expand_dims(y, 1)

    # Divide il dataset in set di training e test.
    # Il parametro test_size determina la proporzione del dataset da utilizzare come test.
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Se richiesto, divide ulteriormente il set di test per creare un set di validazione.
    if validation:
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=random_state)
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        return x_train, x_test, y_train, y_test

def load_cup_test(scaler: MinMaxScaler = None):
    """
    Carica il dataset di test MLCUP23 per la Machine Learning Cup 23.

    Args:
    - scaler (MinMaxScaler, opzionale): Uno scaler per normalizzare i dati. Se fornito,
      verrà utilizzato per trasformare le caratteristiche 'tx' e 'ty'. Default: None.

    Returns:
    - x (ndarray): Matrice delle caratteristiche di input.
    - y (ndarray): Matrice delle etichette/target corrispondenti.
    """

    # Carica il dataset di test CUP da un file CSV.
    # I commenti nel file iniziano con '#', e l'indice del dataframe è impostato sull'ID.
    df = pd.read_csv(
        r"C:\Users\anton\OneDrive\Desktop\X27 - UNIVERSITA'\43 - MACHINE LEARNING\MLproject\data\MLCUP23\MLCUP23.test",
        comment="#",
        index_col="id",
        skipinitialspace=True,
    )

    # Se uno scaler è fornito, applica la trasformazione alle colonne 'tx' e 'ty'.
    if scaler is not None:
        df[["tx", "ty"]] = scaler.transform(df[["tx", "ty"]])

    # Estrae la matrice delle caratteristiche 'x' e le etichette 'y',
    # espandendo le dimensioni per adattarle alle aspettative di alcuni modelli di machine learning.
    x = np.expand_dims(df.drop(["ty", "tx"], axis=1).values, 2)
    y = np.expand_dims(df[["tx", "ty"]].values, 2)

    return x, y


def load_cup_blind_test(scaler: MinMaxScaler = None):
    """
    Carica il dataset senza etichette di test MLCUP23 per la Machine Learning Cup 23.

    Args:
    - scaler (MinMaxScaler, opzionale): Uno scaler per normalizzare i dati. Se fornito,
      verrà utilizzato per trasformare le caratteristiche 'tx' e 'ty'. Default: None.

    Returns:
    - x (ndarray): Matrice delle caratteristiche di input.
    """

    # Carica il dataset di test CUP da un file CSV.
    # I commenti nel file iniziano con '#', e l'indice del dataframe è impostato sull'ID.
    df = pd.read_csv(
        r"C:\Users\anton\OneDrive\Desktop\X27 - UNIVERSITA'\43 - MACHINE LEARNING\MLproject\data\MLCUP23\MLCUP23.test",
        comment="#",
        index_col="id",
        skipinitialspace=True,
    )

    # Applica lo scaler alle caratteristiche, se fornito.
    if scaler is not None:
        df = scaler.transform(df)

    # Estrae la matrice delle caratteristiche 'x', espandendo le dimensioni per
    # adattarle alle aspettative di alcuni modelli di machine learning.
    x = np.expand_dims(df.values, 2)

    return x

def parse_results(results: dict) -> pd.DataFrame:
    """
    Trasforma un dizionario di risultati in un DataFrame per un'analisi più facile.

    Args:
    - results (dict): Un dizionario dove ogni chiave corrisponde a una configurazione sperimentale
      e ogni valore è una tupla o un elenco contenente configurazioni di modello e metriche di performance.

    Returns:
    - pd.DataFrame: Un DataFrame contenente i risultati analizzati per una facile visualizzazione e analisi.
    """

    # Estrae i valori dal dizionario dei risultati. Ogni valore è previsto essere una tupla o un elenco
    # dove il primo elemento è un dizionario di configurazioni e il secondo elemento è un dizionario di metriche.
    res = [val for val in results.values()]

    # Inizializza un DataFrame vuoto per ospitare i risultati analizzati.
    results_df = pd.DataFrame({}, columns=[])

    # Estrae le configurazioni del modello e le metriche di performance dai valori
    # e li assegna a colonne corrispondenti nel DataFrame.
    results_df["eta"] = [k[0]["eta"] for k in res]         # Estrae e assegna il valore 'eta'.
    results_df["nesterov"] = [k[0]["nesterov"] for k in res]  # Estrae e assegna il valore 'nesterov'.
    results_df["reg_type"] = [k[0]["reg_type"] for k in res]  # Estrae e assegna il tipo di regolarizzazione.
    results_df["reg_val"] = [k[0]["reg_val"] for k in res]    # Estrae e assegna il valore di regolarizzazione.
    results_df["tr_mee"] = [k[1]["tr_mee"] for k in res]      # Estrae e assegna l'errore medio di training.
    results_df["val_mee"] = [k[1]["val_mee"] for k in res]    # Estrae e assegna l'errore medio di validazione.
    results_df["loss"] = [k[1]["losses"] for k in res]        # Estrae e assegna i valori delle perdite di training.
    results_df["val_loss"] = [k[1]["val_losses"] for k in res] # Estrae e assegna i valori delle perdite di validazione.

    # Restituisce il DataFrame completato.
    return results_df