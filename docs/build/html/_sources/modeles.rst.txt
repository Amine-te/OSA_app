Modèles Préentraînés
====================

Le système utilise deux modèles de détection préentraînés basés sur YOLOv8 ou une architecture similaire :

void.pt
-------

Modèle dédié à l'identification des espaces vides.

**Caractéristiques :**

- Détection précise des emplacements sans produit.
- Différenciation entre espace vide intentionnel et rupture de stock.
- Prise en compte des étiquettes de prix et des séparateurs.

individual_products.pt
-----------------------

Modèle spécialisé dans la détection des produits individuels (SKU - Stock Keeping Unit).

**Caractéristiques :**

- Identification fine des références précises de produits.
- Capacité à distinguer les variantes d’un même produit (taille, parfum, format).
- Utile pour les tâches de réassort automatique ou de vérification de planogramme.
- Haute précision pour des cas de classification détaillée dans des environnements complexes.

