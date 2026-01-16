import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from generate_data import scale_inputs


class OptionPricingDataset(Dataset):
    def __init__(
        self,
        ticker="^STOXX50E",
        start_date="2015-01-01",
        end_date=None,
        strike=3500.0,
        vol=0.2,
        rate=0.05,
        maturity=1.0,
    ):

        self.df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            multi_level_index=False,
        )

        # Données brutes
        self.prices = self.df["Close"].to_numpy().flatten().astype(np.float32)
        self.dates = self.df.index  # On garde l'index DateTime pour filtrer si besoin

        self.strike = strike
        self.vol = vol
        self.rate = rate
        self.maturity = maturity
        self.s_min = min(self.prices)
        self.s_max = max(self.prices)

    def __len__(self):
        return len(self.prices)

    def call_BS(self, s0):
        # Ta logique Black-Scholes corrigée
        d1 = (
            np.log(s0 / self.strike) + (self.rate + 0.5 * self.vol**2) * self.maturity
        ) / (self.vol * np.sqrt(self.maturity))
        d2 = d1 - self.vol * np.sqrt(self.maturity)

        t_d1 = torch.tensor(d1, dtype=torch.float32)
        t_d2 = torch.tensor(d2, dtype=torch.float32)

        price = s0 * torch.special.ndtr(t_d1) - self.strike * np.exp(
            -self.rate * self.maturity
        ) * torch.special.ndtr(t_d2)
        return price / self.strike  # Prix normalisé

    def __getitem__(self, idx):
        s0 = self.prices[idx]

        # 1. Création des FEATURES (X)
        # Note: On ne met PAS la date "string" dans le tenseur d'entraînement, le modèle ne peut pas lire "2020-01-01"
        x = torch.tensor(
            [s0 / self.strike, self.maturity], dtype=torch.float32  # Moneyness
        )

        # 2. Création du LABEL (y)
        y = torch.tensor([self.call_BS(s0)], dtype=torch.float32)

        return x, y


# --- FONCTION POUR GÉNÉRER LES 4 DATASETS ---
def get_train_test_datasets(dataset, split_ratio=0.8):
    """
    Coupe le dataset en 4 tenseurs de manière temporelle (pas de mélange aléatoire).
    """
    total_len = len(dataset)
    train_size = int(total_len * split_ratio)

    # On récupère toutes les données d'un coup pour les slicer
    # (C'est rapide car dataset[i] calcule à la volée, on boucle sur tout)
    all_X = []
    all_y = []

    print("Génération des données en cours...")
    for i in range(total_len):
        x, y = dataset[i]
        all_X.append(x)
        all_y.append(y)

    # On empile le tout en deux gros tenseurs
    X_full = torch.stack(all_X)
    y_full = torch.stack(all_y)

    # Slicing Temporel (Les 80% premiers jours vs les 20% derniers jours)
    X_train = X_full[:train_size]
    y_train = y_full[:train_size]

    X_test = X_full[train_size:]
    y_test = y_full[train_size:]

    return X_train, y_train, X_test, y_test


# --- UTILISATION ---
if __name__ == "__main__":
    # 1. Instancier le dataset global
    full_dataset = OptionPricingDataset(ticker="^STOXX50E", start_date="2015-01-01")

    # 2. Récupérer les 4 blocs séparés
    X_train, y_train, X_test, y_test = get_train_test_datasets(
        full_dataset, split_ratio=0.8
    )

    print("-" * 30)
    print(f"Total jours : {len(full_dataset)}")
    print(f"X_train shape : {X_train.shape}  (Features d'entraînement)")
    print(f"y_train shape : {y_train.shape}  (Prix Call d'entraînement)")
    print(f"X_test shape  : {X_test.shape}   (Features de test)")
    print(f"y_test shape  : {y_test.shape}   (Prix Call de test)")
    print("-" * 30)

    # 3. Créer des DataLoaders prêts pour PyTorch
    # C'est souvent plus pratique d'avoir des loaders que des tenseurs bruts
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=32, shuffle=False
    )

    print(X_train)
    print(y_train)
    plt.plot(X_train[:, 0], y_train)
    plt.grid(True)
    plt.show()
    # Vérification d'un batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch X: \n{x_batch[0]}")  # [Moneyness, Vol, Rate, T]
    print(f"Batch y: \n{y_batch[0]}")  # [Prix Call]
