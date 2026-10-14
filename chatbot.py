import re
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random


#  CORPUS


paragraphes = [
    [
        "Le chat dort sur le canapé.",
        "Il se réveille quand le soleil apparaît.",
        "Puis il regarde les oiseaux dans le jardin."
    ],
    [
       "Le chien joue dans le jardin.",
        "Il court après une balle.",
        "Puis il retourne dormir sur le canapé."
    ],
    [
        "Le soleil se lève dans le ciel.",
        "La lumière éclaire la maison.",
        "Le chat se réveille et cherche son repas."
    ],
    [
        "Un oiseau chante dans un arbre.",
        "Le chat observe l'oiseau depuis la fenêtre.",
        "L'oiseau s'envole vers le jardin."
    ],
    [
        "Le garçon prend un livre.",
        "Il s'assoit dans le jardin.",
        "Puis il commence à lire tranquillement."
    ],
    [
        "La pluie tombe sur le jardin.",
        "Le chien rentre dans la maison.",
        "Il s'allonge près de la fenêtre."
    ] ]

# séparation

random.shuffle(paragraphes)
nombre_train = int(len(paragraphes)*0.8)
paragraphes = paragraphes[:nombre_train]
paragraphes_val = paragraphes[nombre_train:]


#  TOKENISATION


tokens = []

for paragraphe in paragraphes:
    for phrase in paragraphe:
        mots = re.findall(r"\b\w+\b", phrase.lower())
        tokens.extend(mots)
        tokens.append("<EOS>")

vocabulaire = sorted(set(tokens))

mot_vers_index = {
    mot: i for i, mot in enumerate(vocabulaire)
}

index_vers_mot = {
    i: mot for i, mot in enumerate(vocabulaire)
}

tokens_numeriques = [
    mot_vers_index[mot] for mot in tokens
]

index_eos = mot_vers_index["<EOS>"]

print("Nombre de tokens :", len(tokens))
print("Taille du vocabulaire :", len(vocabulaire))
print("Exemple :", tokens[:20])

# 4. TRANSFORMATION DES MOTS EN NOMBRES

tokens_numeriques = [
    mot_vers_index[mot]
    for mot in tokens
]

print()
print("Tokens numériques :")
print(tokens_numeriques)


# préparer les sequences

def preparer_donnees(paragraphes, contexte, mot_vers_index):
    X, Y = [], []

    for paragraphe in paragraphes:
        mots = re.findall(r"\b\w+\b", paragraphe.lower())
        mots.append("<EOS>")

        # Ignorer les phrases contenant des mots inconnus
        if any(mot not in mot_vers_index for mot in mots):
            continue

        indices = [mot_vers_index[mot] for mot in mots]

        # Ajouter un contexte de début pour les phrases courtes
        for i in range(len(indices)):
            debut = max(0, i - contexte)
            entree = indices[debut:i]
            cible = indices[i]

            # Compléter à gauche pour garder une longueur fixe
            entree = [mot_vers_index["<PAD>"]] * (
                contexte - len(entree)
            ) + entree

            X.append(entree)
            Y.append(cible)

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long)
    )


#  CREATION DES EXEMPLES D'ENTRAINEMENT

contexte = 12


import random
import re
import torch

# 1. Séparer les phrases
random.seed(42)
paragraphes_melangees = paragraphes.copy()
random.shuffle(paragraphes_melangees)

nombre_train = int(0.8 * len(paragraphes_melangees))
paragraphes_train = paragraphes_melangees[:nombre_train]
paragraphes_eval = paragraphes_melangees[nombre_train:]

# 2. Construire le vocabulaire à partir du corpus d'entraînement
tokens_train = []

for phrase in paragraphes_train:
    mots = re.findall(r"\b\w+\b", phrase.lower())
    tokens_train.extend(mots)
    tokens_train.append("<EOS>")

vocabulaire = sorted(set(tokens_train))
vocabulaire.extend(["<PAD>", "<UNK>"])

mot_vers_index = {
    mot: i for i, mot in enumerate(vocabulaire)
}
index_vers_mot = {
    i: mot for mot, i in mot_vers_index.items()
}

# 3. Préparer les séquences


def preparer_donnees(liste_paragraphes, contexte, mot_vers_index):
    X, Y = [], []

    for phrase in liste_paragraphes:
        mots = re.findall(r"\b\w+\b", phrase.lower())
        mots.append("<EOS>")

        # Remplacer les mots inconnus
        indices = [
            mot_vers_index.get(mot, mot_vers_index["<UNK>"])
            for mot in mots
        ]

        # Chaque exemple prédit le mot suivant
        for i in range(len(indices)):
            entree = indices[max(0, i - contexte):i]
            cible = indices[i]

            entree = (
                [mot_vers_index["<PAD>"]] * (contexte - len(entree))
                + entree
            )

            X.append(entree)
            Y.append(cible)

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long)
    )

# 4. Créer les quatre variables
X_train, Y_train = preparer_donnees(
    paragraphes_train, contexte, mot_vers_index
)

X_eval, Y_eval = preparer_donnees(
    paragraphes_eval, contexte, mot_vers_index
)

# 5. Vérifier les dimensions
print("X_train :", X_train.shape)
print("Y_train :", Y_train.shape)
print("X_eval :", X_eval.shape)
print("Y_eval :", Y_eval.shape)
print("Taille du vocabulaire :", len(vocabulaire))
#  ENCODAGE POSITIONNEL

def positional_encoding(longueur, dimension):

    PE = torch.zeros(
        longueur,
        dimension
    )

    for pos in range(longueur):

        for i in range(
            0,
            dimension,
            2
        ):

            PE[pos, i] = math.sin(
                pos / (
                    10000 ** (i / dimension)
                )
            )

            if i + 1 < dimension:

                PE[pos, i + 1] = math.cos(
                    pos / (
                        10000 ** (i / dimension)
                    )
                )

    return PE


#


#  BLOC TRANSFORMER


class MiniTransformerBlock(nn.Module):
    def __init__(self, dimension, contexte, nombre_tetes=4):
        super().__init__()

        # Attention avec plusieurs têtes
        self.attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=nombre_tetes,
            batch_first=True
        )

        # Réseau de neurones appliqué à chaque token
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.ReLU(),
            nn.Linear(dimension * 4, dimension)
        )

        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)


        self.register_buffer(
            "masque_causal",
            torch.triu(
                torch.ones(contexte, contexte, dtype=torch.bool),
                diagonal=1
            )
        )

    def forward(self, x):
        longueur = x.size(1)

        masque = self.masque_causal[:longueur, :longueur]

        # Self-attention causale
        sortie_attention, _ = self.attention(
            x, x, x,
            attn_mask=masque,
            need_weights=False
        )

        # Connexion résiduelle et normalisation
        x = self.norm1(x + sortie_attention)

        # Réseau feed-forward
        sortie_ffn = self.ffn(x)

        # Deuxième connexion résiduelle et normalisation
        x = self.norm2(x + sortie_ffn)

        return x




# MINIGPT


class MiniGPT(nn.Module):
    def __init__(
        self,
        taille_vocabulaire,
        dimension=32,
        nombre_blocs=2,
        contexte=3,
        nombre_tetes=4
    ):
        super().__init__()

        self.contexte = contexte

        # Transformer les identifiants en vecteurs
        self.embedding = nn.Embedding(
            taille_vocabulaire,
            dimension
        )

        # Ajouter l'information de position
        self.register_buffer(
            "position",
            positional_encoding(contexte, dimension)
        )

        # Empiler plusieurs blocs Transformer
        self.transformers = nn.Sequential(
            *[
                MiniTransformerBlock(
                    dimension,
                    contexte,
                    nombre_tetes
                )
                for _ in range(nombre_blocs)
            ]
        )

        # Produire un score pour chaque token du vocabulaire
        self.output = nn.Linear(
            dimension,
            taille_vocabulaire
        )

    def forward(self, tokens):
        # tokens : (batch, longueur)
        x = self.embedding(tokens)

        # Ajouter les positions
        x = x + self.position[:x.size(1)]

        # Passer dans les blocs Transformer
        x = self.transformers(x)

        # Garder la représentation du dernier token
        x = x[:, -1, :]

        # Retourner les logits
        logits = self.output(x)

        return logits


# CREATION DU MODELE

model = MiniGPT(
    taille_vocabulaire=len(vocabulaire),
    dimension=32,
    nombre_blocs=2,
    contexte=contexte,
    nombre_tetes=4
)

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003
)

for epoch in range(2000):
    # Entraînement
    model.train()

    prediction = model(X_train)
    loss_train = loss_function(prediction, Y_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # Validation : aucune mise à jour des paramètres
    model.eval()

    with torch.no_grad():
        prediction_val = model(X_val)
        loss_val = loss_function(prediction_val, Y_val)

    if (epoch + 1) % 100 == 0:
        print(
            f"Époque {epoch + 1:4d} | "
            f"Train : {loss_train.item():.4f} | "
            f"Val : {loss_val.item():.4f}"
        )


#  OPTIMISEUR

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.005
)


#  ENTRAINEMENT

epochs = 2000

embeddings_avant = (
    model.embedding.weight
    .detach()
    .clone()
)

for epoch in range(epochs):

    model.train()


    # Prédiction

    prediction = model(X)


    # Calcul de l'erreur

    loss = loss_function(
        prediction,
        Y
    )


    # Remise à zéro des gradients

    optimizer.zero_grad()


    # Backpropagation

    loss.backward()


    # Mise à jour des paramètres

    optimizer.step()


    # Affichage

    if epoch % 200 == 0:

        print(
            f"Époque {epoch} | "
            f"Loss = {loss.item():.4f}"
        )

embeddings_apres = (
    model.embedding.weight
    .detach()
    .clone()
)

def probabilites_suivantes(texte, nombre=5):

    model.eval()

    tokens = re.findall(
        r"\b\w+\b",
        texte.lower()
    )

    tokens = tokens[-contexte:]

    indices = [
        mot_vers_index[mot]
        for mot in tokens
    ]

    entree = torch.tensor(
        [indices],
        dtype=torch.long
    )

    with torch.no_grad():

        logits = model(entree)

        probabilites = torch.softmax(
            logits,
            dim=-1
        )

    valeurs, indices = torch.topk(
        probabilites,
        nombre
    )

    print()
    print(
        f'"{texte}" :' )

    for probabilite, indice in zip(
        valeurs[0],
        indices[0]
    ):

        mot = index_vers_mot[
            indice.item()
        ]

        print(
            f"{mot:15s} "

        )

#  FONCTION DE PREDICTION

def predire(texte):

    model.eval()


    # Tokenisation

    tokens = re.findall(
        r"\b\w+\b",
        texte.lower()
    )


    # contexte maximal

    tokens = tokens[
        -contexte:
    ]


    #les mots existent

    for mot in tokens:

        if mot not in mot_vers_index:

            raise ValueError(
                f"Le mot '{mot}' "
                f"n'est pas dans le vocabulaire."
            )


    # Conversion

    indices = [

        mot_vers_index[mot]

        for mot in tokens
    ]


    entree = torch.tensor(
        [indices],
        dtype=torch.long
    )


    # Prédiction

    with torch.no_grad():

        logits = model(
            entree
        )

        prediction = torch.argmax(
            logits,
            dim=-1
        ).item()


    return index_vers_mot[
        prediction
    ]




#  GENERATION DE TEXTE


import torch
import torch.nn.functional as F
import re


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

    model.eval()

    # Convertir le texte de départ en tokens
    mots = re.findall(r"\b\w+\b", debut.lower())

    if not mots:
        raise ValueError("Le texte de départ est vide.")

    # Le modèle ne peut utiliser que les mots connus
    inconnus = [mot for mot in mots if mot not in mot_vers_index]

    if inconnus:
        raise ValueError(f"Tokens inconnus : {inconnus}")

    resultat = mots.copy()
    eos = mot_vers_index["<EOS>"]

    for _ in range(longueur_max):
        # Garder uniquement les derniers tokens du contexte
        contexte_mots = resultat[-contexte:]

        indices = [
            mot_vers_index[mot]
            for mot in contexte_mots
        ]

        entree = torch.tensor(
            [indices],
            dtype=torch.long
        )

        with torch.no_grad():
            logits = model(entree)

            # Contrôler la diversité des prédictions
            logits = logits / temperature

            # Éviter de choisir EOS immédiatement
            # si aucun mot n'a encore été généré
            if len(resultat) == len(mots):
                logits[0, eos] = float("-inf")

            # Garder seulement les top_k meilleurs choix
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                valeurs, indices_top = torch.topk(logits, k)

                logits_filtres = torch.full_like(
                    logits,
                    float("-inf")
                )

                logits_filtres.scatter_(
                    1,
                    indices_top,
                    valeurs
                )

                logits = logits_filtres

            probabilites = F.softmax(logits, dim=-1)

            prochain_index = torch.multinomial(
                probabilites,
                num_samples=1
            ).item()

        # Arrêter à la fin de la phrase
        if prochain_index == eos:
            break

        prochain_mot = index_vers_mot[prochain_index]
        resultat.append(prochain_mot)

    phrase = " ".join(resultat)

    # Ajouter une majuscule et une ponctuation simple
    phrase = phrase[0].upper() + phrase[1:]

    if phrase[-1] not in ".!?":
        phrase += "."

    return phrase

#  GENERATION

phrase = generer_phrase(
        debut="le chat ",
        temperature=0.9,
        top_k=5,
        longueur_max=20
    )

print(f"{i + 1}. {phrase}")

#=======================================

embeddings = model.embedding.weight.detach().numpy()

pca = PCA(
    n_components=2
)

embeddings_2d = pca.fit_transform(
    embeddings
)

plt.figure(figsize=(10, 8))

for i, mot in index_vers_mot.items():

    x = embeddings_2d[i, 0]
    y = embeddings_2d[i, 1]

    plt.scatter(x, y)

    plt.text(
        x + 0.02,
        y + 0.02,
        mot
    )

plt.title("Projection des embeddings")
plt.xlabel("Dimension principale 1")
plt.ylabel("Dimension principale 2")
plt.grid()

plt.show()


indice = mot_vers_index["chat"]

avant = embeddings_avant[indice]
apres = embeddings_apres[indice]

deplacement = torch.norm(
    apres - avant
)

print(
    "Déplacement de l'embedding de chat :",
    deplacement.item()
)



#=======================================


