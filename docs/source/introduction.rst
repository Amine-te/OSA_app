Introduction
=================================================


Le système de détection On-Shelf Availability (OSA) est une solution avancée de vision par ordinateur conçue pour identifier automatiquement les espaces vides sur les rayons, détecter les étagères et reconnaître les produits présents dans un environnement commercial.

Dans le secteur de la grande distribution et du commerce de détail, les ruptures de stock représentent un enjeu majeur pouvant entraîner jusqu'à 4% de pertes de ventes annuelles. Notre solution technologique permet aux détaillants d'optimiser la gestion des stocks, d'améliorer l'expérience client et de maximiser les ventes en assurant une disponibilité optimale des produits.

Le système OSA utilise des algorithmes d'intelligence artificielle de pointe pour analyser en temps réel l'état des rayons et fournir des informations précieuses aux gestionnaires de magasins.

Objectifs du Système
--------------------

Le système a été développé avec des objectifs précis pour répondre aux besoins spécifiques du secteur de la distribution :

- **Détecter les espaces vides sur les étagères** : Identification automatique des zones nécessitant un réapprovisionnement
- **Identifier et localiser les étagères** : Cartographie précise de l'espace commercial pour un suivi optimal
- **Reconnaître les produits présents** : Classification des articles présents sur les rayons
- **Détecter les produits individuels** : Comptage précis des unités pour une gestion fine des stocks
- **Générer des rapports en temps réel** : Documentation détaillée sur l'état des stocks pour faciliter la prise de décision
- **Alerter le personnel** : Notifications automatiques en cas de rupture de stock pour une action rapide

Architecture du Système
-----------------------

Le système OSA repose sur une architecture modulaire comprenant quatre modèles de détection principaux qui fonctionnent en séquence pour assurer une analyse complète et précise des rayonnages.

.. code-block::

    ┌─────────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌───────────────────────────┐
    │                 │     │                │     │                 │     │                           │
    │  Détection des  │────▶│ Détection des  │────▶│ Détection des   │────▶│ Détection des produits    │
    │    étagères     │     │     vides      │     │    produits     │     │     individuels           │
    │   (shelf.pt)    │     │   (void.pt)    │     │  (products.pt)  │     │ (individual_products.pt)  │
    │                 │     │                │     │                 │     │                           │
    └─────────────────┘     └────────────────┘     └─────────────────┘     └───────────────────────────┘
             │                     │                       │                           │
             ▼                     ▼                       ▼                           ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                   Traitement des données                                      │
    └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                 Génération des rapports                                       │
    └───────────────────────────────────────────────────────────────────────────────────────────────┘

Description des composants
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Détection des étagères (shelf.pt)**
   - Localise et délimite les structures d'étagères dans l'environnement commercial
   - Utilise un modèle YOLOv8 optimisé pour la reconnaissance de structures commerciales
   - Fournit les coordonnées précises pour les analyses subséquentes

2. **Détection des vides (void.pt)**
   - Analyse les sections d'étagères identifiées pour repérer les espaces vides
   - Emploie un algorithme de segmentation sémantique pour différencier les zones occupées et inoccupées
   - Calcule le pourcentage d'espace vide par rayon

3. **Détection des produits (products.pt)**
   - Identifie les catégories de produits présents sur les rayons
   - Utilise un réseau de neurones convolutif entraîné sur une vaste base de données d'articles commerciaux
   - Permet d'analyser la répartition des produits par catégorie

4. **Détection des produits individuels (individual_products.pt)**
   - Compte avec précision les unités individuelles de chaque produit
   - Emploie des techniques de vision par ordinateur avancées pour distinguer les articles même en cas de chevauchement
   - Fournit des données granulaires pour un inventaire précis

5. **Traitement des données**
   - Agrège et analyse les informations collectées par les différents modèles
   - Applique des algorithmes de data mining pour extraire des insights pertinents
   - Stocke les données structurées pour une exploitation ultérieure

6. **Génération des rapports**
   - Produit des rapports détaillés et personnalisables selon les besoins des utilisateurs
   - Intègre des visualisations intuitives pour faciliter l'interprétation des données
   - Permet l'export dans différents formats (PDF, CSV, JSON) pour une intégration avec d'autres systèmes

Cas d'Utilisation
-----------------

Notre système OSA s'adapte à différents scénarios d'utilisation dans le secteur de la distribution :

- **Surveillance continue des rayons en magasin**
  - Monitoring en temps réel de l'état des rayons
  - Détection immédiate des anomalies ou des ruptures
  - Suivi des tendances de consommation au cours de la journée

- **Audit de merchandising**
  - Évaluation précise du placement et de la présentation des produits
  - Vérification de la conformité aux planogrammes
  - Analyse de l'efficacité des stratégies de mise en rayon

- **Optimisation de la chaîne d'approvisionnement**
  - Anticipation des besoins de réapprovisionnement
  - Réduction des délais entre la détection d'une rupture et son traitement
  - Synchronisation avec les systèmes de gestion d'inventaire

- **Analyse des performances commerciales**
  - Corrélation entre la disponibilité des produits et les ventes
  - Identification des produits à forte rotation
  - Évaluation de l'impact des promotions sur l'état des stocks

Avantages et Impact
-------------------

L'implémentation de notre système OSA offre de nombreux avantages mesurables pour les commerces de détail :

- **Réduction des pertes de ventes** : Diminution de 30% à 40% des situations de rupture de stock, se traduisant par une augmentation directe du chiffre d'affaires
- **Amélioration de l'expérience client** : Satisfaction accrue des clients grâce à une disponibilité constante des produits recherchés
- **Optimisation des processus** : Réduction de 50% du temps consacré aux contrôles manuels des stocks par le personnel
- **Insights stratégiques** : Collecte de données précieuses sur les comportements d'achat et les tendances de consommation
- **ROI rapide** : Retour sur investissement généralement observé dans les 6 à 12 mois suivant l'implémentation

Notre solution s'intègre facilement dans l'infrastructure existante grâce à ses interfaces API standardisées et sa conception modulaire. Elle est évolutive et peut s'adapter aux besoins spécifiques de chaque client, qu'il s'agisse d'une petite surface de vente ou d'un hypermarché.


Technologies Utilisées
-----------------------

Le système OSA s'appuie sur un ensemble de technologies de pointe pour offrir des performances optimales :

- **Frameworks d'IA** : PyTorch, TensorFlow
- **Vision par ordinateur** : OpenCV, YOLOv8,YOLOv11,DTER,CNN
- **Traitement des données** : Python, Pandas, NumPy,Matplotlib,
- **Interface utilisateur** : Steamlit
- **Déploiement** : 
- **Stockage** : 

Ces technologies ont été sélectionnées pour leur robustesse, leur capacité à traiter des données en temps réel et leur facilité d'intégration dans des environnements commerciaux existants.
 
