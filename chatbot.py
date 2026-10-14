import copy
import math
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


# CORPUS ET PARAMETRES

random.seed(42)
torch.manual_seed(42)

contexte = 12
dimension = 32
nombre_blocs = 2
nombre_tetes = 4
taux_apprentissage = 0.005
nombre_epochs_max = 2000
patience = 100

paragraphes = [
    "Le chat mange une souris.",
    "Le chat dort sur le canapé.",
    "Le chat joue dans le jardin.",
    "Le chat regarde la lune.",
    "Le chat observe un oiseau.",
    "Le chien mange une croquette.",
    "Le chien joue dans le jardin.",
    "Le chien dort sur le canapé.",
    "Le chien regarde le soleil.",
    "Le chien poursuit une balle.",
    "Le soleil brille dans le ciel.",
    "Le soleil éclaire la maison.",
    "Le soleil chauffe le jardin.",
    "La lune apparaît dans le ciel.",
    "La lune éclaire la nuit.",
    "La lune brille dans le ciel.",
    "Un oiseau vole dans le ciel.",
    "Un oiseau chante dans le jardin.",
    "Un oiseau regarde le chat.",
    "Une souris mange une graine.",
    "Une souris court dans le jardin.",
    "Une souris dort dans une maison.",
    "Le garçon joue dans le jardin.",
    "Le garçon regarde le chien.",
    "Le garçon mange une pomme.",
    "La fille joue avec le chat.",
    "La fille regarde la lune.",
    "La fille mange une pomme."
]

# SEPARER LES PARAGRAPHES EN ENTRAINEMENT ET EVALUATION

paragraphes_melanges = paragraphes.copy()
random.shuffle(paragraphes_melanges)

if len(paragraphes_melanges) < 2:
    raise ValueError("Il faut au moins deux paragraphes pour séparer entraînement et évaluation.")

nombre_train = max(1, int(0.8 * len(paragraphes_melanges)))
nombre_train = min(nombre_train, len(paragraphes_melanges) - 1)

paragraphes_train = paragraphes_melanges[:nombre_train]
paragraphes_eval = paragraphes_melanges[nombre_train:]


#  TOKENISATION ET VOCABULAIRE

def decouper_texte(texte):
    """Découpe le texte en mots simples, en minuscules."""
    return re.findall(r"\b\w+\b", texte.lower())


tokens_train = []
for paragraphe in paragraphes_train:
    tokens_train.extend(decouper_texte(paragraphe))
    tokens_train.append("<EOS>")

vocabulaire = sorted(set(tokens_train))
vocabulaire.extend(["<PAD>", "<UNK>"])

mot_vers_index = {mot: i for i, mot in enumerate(vocabulaire)}
index_vers_mot = {i: mot for mot, i in mot_vers_index.items()}

taille_vocabulaire = len(vocabulaire)

print("Nombre de paragraphes d'entraînement :", len(paragraphes_train))
print("Nombre de paragraphes d'évaluation :", len(paragraphes_eval))
print("Taille du vocabulaire :", taille_vocabulaire)


# PREPARER LES SEQUENCES X ET LES CIBLES Y

def preparer_donnees(liste_paragraphes, contexte, mot_vers_index):
    X, Y = [], []
    index_pad = mot_vers_index["<PAD>"]
    index_unk = mot_vers_index["<UNK>"]

    for paragraphe in liste_paragraphes:
        mots = decouper_texte(paragraphe)
        mots.append("<EOS>")

        indices = [
            mot_vers_index.get(mot, index_unk)
            for mot in mots
        ]

        for i, cible in enumerate(indices):
            debut = max(0, i - contexte)
            entree = indices[debut:i]
            entree = [index_pad] * (contexte - len(entree)) + entree

            X.append(entree)
            Y.append(cible)

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long)
    )


X_train, Y_train = preparer_donnees(
    paragraphes_train, contexte, mot_vers_index
)
X_eval, Y_eval = preparer_donnees(
    paragraphes_eval, contexte, mot_vers_index
)

print("Forme de X_train :", tuple(X_train.shape))
print("Forme de Y_train :", tuple(Y_train.shape))
print("Forme de X_eval :", tuple(X_eval.shape))
print("Forme de Y_eval :", tuple(Y_eval.shape))


#  ENCODAGE POSITIONNEL


def positional_encoding(contexte, dimension):
    position = torch.arange(contexte, dtype=torch.float32).unsqueeze(1)
    indices = torch.arange(0, dimension, 2, dtype=torch.float32)

    frequences = torch.exp(
        -math.log(10000.0) * indices / dimension
    )

    encodage = torch.zeros(contexte, dimension)
    encodage[:, 0::2] = torch.sin(position * frequences)

    nombre_colonnes_cos = encodage[:, 1::2].shape[1]
    encodage[:, 1::2] = torch.cos(
        position * frequences[:nombre_colonnes_cos]
    )

    return encodage


# BLOC TRANSFORMER AVEC ATTENTION MULTI-TETES CAUSALE

class MiniTransformerBlock(nn.Module):
    def __init__(self, dimension, contexte, nombre_tetes=4):
        super().__init__()

        if dimension % nombre_tetes != 0:
            raise ValueError(
                "dimension doit être divisible par nombre_tetes."
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=nombre_tetes,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.ReLU(),
            nn.Linear(dimension * 4, dimension)
        )

        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)

        masque = torch.triu(
            torch.ones(contexte, contexte, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("masque_causal", masque)

    def forward(self, x):
        longueur = x.size(1)
        masque = self.masque_causal[:longueur, :longueur]

        sortie_attention, _ = self.attention(
            x, x, x,
            attn_mask=masque,
            need_weights=False
        )

        x = self.norm1(x + sortie_attention)
        sortie_ffn = self.ffn(x)
        x = self.norm2(x + sortie_ffn)

        return x


#  MODELE MINIGPT

class MiniGPT(nn.Module):
    def __init__(
        self,
        taille_vocabulaire,
        dimension=32,
        nombre_blocs=2,
        contexte=12,
        nombre_tetes=4
    ):
        super().__init__()

        self.contexte = contexte

        self.embedding = nn.Embedding(
            taille_vocabulaire,
            dimension,
            padding_idx=None
        )

        self.register_buffer(
            "position",
            positional_encoding(contexte, dimension)
        )

        self.transformers = nn.Sequential(*[
            MiniTransformerBlock(dimension, contexte, nombre_tetes)
            for _ in range(nombre_blocs)
        ])

        self.output = nn.Linear(dimension, taille_vocabulaire)

    def forward(self, tokens):
        x = self.embedding(tokens)
        x = x + self.position[:x.size(1)]
        x = self.transformers(x)

        x = x[:, -1, :]
        logits = self.output(x)

        return logits


# CREER LE MODELE ET LES OUTILS D'ENTRAINEMENT

model = MiniGPT(
    taille_vocabulaire=taille_vocabulaire,
    dimension=dimension,
    nombre_blocs=nombre_blocs,
    contexte=contexte,
    nombre_tetes=nombre_tetes
)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=taux_apprentissage
)


# ENTRAINEMENT AVEC EVALUATION ET EARLY STOPPING

meilleure_loss_eval = float("inf")
meilleur_etat = None
epochs_sans_amelioration = 0
min_delta = 1e-4

for epoch in range(nombre_epochs_max):
    model.train()

    prediction = model(X_train)
    loss_train = loss_function(prediction, Y_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction_eval = model(X_eval)
        loss_eval = loss_function(prediction_eval, Y_eval)

    valeur_eval = loss_eval.item()

    if valeur_eval < meilleure_loss_eval - min_delta:
        meilleure_loss_eval = valeur_eval
        meilleur_etat = copy.deepcopy(model.state_dict())
        epochs_sans_amelioration = 0
    else:
        epochs_sans_amelioration += 1

    if (epoch + 1) % 100 == 0:
        print(
            f"Époque {epoch + 1:4d} | "
            f"perte entraînement : {loss_train.item():.4f} | "
            f"perte évaluation : {valeur_eval:.4f}"
        )

    if epochs_sans_amelioration >= patience:
        print(f"Arrêt anticipé à l'époque {epoch + 1}.")
        break

if meilleur_etat is not None:
    model.load_state_dict(meilleur_etat)

print(f"Meilleure perte d'évaluation : {meilleure_loss_eval:.4f}")


#  GENERATION DE TEXTE

def generer_phrase(
    debut="le chat",
    temperature=0.8,
    top_k=5,
    longueur_max=20
):
    if temperature <= 0:
        raise ValueError("La température doit être strictement positive.")
    if longueur_max < 1:
        raise ValueError("La longueur maximale doit être positive.")

    mots_depart = decouper_texte(debut)
    if not mots_depart:
        raise ValueError("Le texte de départ est vide.")

    inconnus = [
        mot for mot in mots_depart
        if mot not in mot_vers_index
    ]
    if inconnus:
        raise ValueError(
            f"Mots inconnus dans le texte de départ : {inconnus}. "
            "Essaie un début utilisant des mots du corpus d'entraînement."
        )

    resultat = mots_depart.copy()
    index_eos = mot_vers_index["<EOS>"]
    model.eval()

    for _ in range(longueur_max):
        mots_contexte = resultat[-contexte:]
        indices = [mot_vers_index[mot] for mot in mots_contexte]

        entree = torch.tensor([indices], dtype=torch.long)

        with torch.no_grad():
            logits = model(entree)
            logits = logits / temperature

            if len(resultat) == len(mots_depart):
                logits[0, index_eos] = float("-inf")

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                valeurs, indices_top = torch.topk(logits, k)

                logits_filtres = torch.full_like(
                    logits, float("-inf")
                )
                logits_filtres.scatter_(1, indices_top, valeurs)
                logits = logits_filtres

            probabilites = F.softmax(logits, dim=-1)
            prochain_index = torch.multinomial(
                probabilites, num_samples=1
            ).item()

        if prochain_index == index_eos:
            break

        prochain_mot = index_vers_mot[prochain_index]

        if prochain_mot in ("<PAD>", "<UNK>"):
            continue

        resultat.append(prochain_mot)

    phrase = " ".join(resultat)
    phrase = phrase[0].upper() + phrase[1:]

    if phrase[-1] not in ".!?":
        phrase += "."

    return phrase



# TESTER LA GENERATION

print("\nQuelques générations :")
for debut in ["le chat", "le chien", "la lune"]:
    try:
        print(f"{debut!r} -> {generer_phrase(debut)}")
    except ValueError as erreur:
        print(f"{debut!r} -> {erreur}")

