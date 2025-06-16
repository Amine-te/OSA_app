==================================================
Guide d'utilisation 2
==================================================

Vue d'ensemble
==============

L'application **Retail Shelf Analysis** est un outil d'analyse intelligent des rayons de magasin utilisant la vision par ordinateur et l'intelligence artificielle. Elle permet d'analyser automatiquement les niveaux de stock, détecter les produits manquants et générer des recommandations de réapprovisionnement.

Fonctionnalités principales
===========================

🔍 Détection de produits
-----------------------

* Détection automatique des produits sur les rayons
* Classification des produits par catégorie  
* Détection des espaces vides (voids)
* Calcul précis des niveaux de stock

📊 Analyse avancée
-----------------

* Analyse des images individuelles
* Analyse vidéo frame par frame
* Analyse de tendances temporelles
* Visualisations interactives

📤 Export des résultats
----------------------

* Export JSON pour l'intégration système
* Export CSV pour l'analyse de données
* Rapports texte détaillés
* Visualisations téléchargeables

Installation et configuration
=============================

Prérequis
---------

.. code-block:: bash

   pip install streamlit opencv-python numpy pillow plotly pandas matplotlib

Structure des fichiers
----------------------

.. code-block:: text

   project/
   ├── app.py                     # Application Streamlit principale
   ├── pipeline.py               # Pipeline d'analyse (EnhancedRetailPipeline)
   ├── notebookClassification.ipynb  # Pour l'entrainemnt du model de classifocation
   ├── models/
   │   ├── individual_products.pt    # Modèle YOLO pour la détection
   │   ├── void.pt                   # Modèle de détection des voids
   │   └── classifier/
   │       ├── best_lightweight_cnn.pth  # Modèle CNN de classification
   │       └── model_info.json          # Informations sur les classes
   └── requirements.txt

Guide d'utilisation
===================

1. Démarrage de l'application
-----------------------------

.. code-block:: bash

   streamlit run app.py

2. Configuration initiale
------------------------

Configuration des modèles
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Panneau latéral** : Accédez aux paramètres de configuration
2. **Chemins des modèles** : Vérifiez ou modifiez les chemins vers vos modèles :

   * Modèle YOLO : ``models/individual_products.pt``
   * Modèle CNN : ``models/classifier/best_lightweight_cnn.pth``
   * Modèle Void : ``models/void.pt``

Configuration des classes de produits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les noms des classes sont automatiquement chargés depuis ``model_info.json``. Si le fichier n'existe pas, vous pouvez les saisir manuellement :

.. code-block:: text

   cocacola,oil,water,juice,soda

Seuils de confiance
~~~~~~~~~~~~~~~~~~

* **Détection de produits** : 0.5 (recommandé)
* **Détection de voids** : 0.5 (recommandé)

3. Initialisation du pipeline
-----------------------------

Cliquez sur **"🚀 Initialize Pipeline"** dans le panneau latéral pour charger les modèles.

4. Analyse d'images
------------------

Upload d'image
~~~~~~~~~~~~~~

1. Dans la section **"📁 Upload Media"**
2. Sélectionnez une image (JPG, PNG, BMP)
3. L'image s'affiche automatiquement

Lancement de l'analyse
~~~~~~~~~~~~~~~~~~~~~~

1. Cliquez sur **"🔍 Analyze Image"**
2. Attendez le traitement (quelques secondes)
3. Les résultats s'affichent automatiquement

Interprétation des résultats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Métriques générales :**

* **Total Products** : Nombre total de produits détectés
* **Missing Products** : Estimation des produits manquants
* **Overall Stock** : Pourcentage de stock global

**Visualisations :**

* **Graphique en barres** : Comparaison stock actuel vs manquant
* **Jauges de stock** : Niveau de stock par produit
* **Tableau détaillé** : Informations complètes par produit

**Statuts de stock :**

* 🟢 **GOOD** (≥90%) : Stock suffisant
* 🟡 **MODERATE** (70-89%) : Stock moyen
* 🔴 **LOW** (<70%) : Stock faible - réapprovisionnement nécessaire

5. Analyse vidéo
---------------

Configuration vidéo
~~~~~~~~~~~~~~~~~~

1. **Frame Interval** : Intervalle d'extraction (secondes)
2. **Max Frames** : Nombre maximum de frames à analyser
3. **Analysis Mode** : Mode d'analyse

   * **Single Frame** : Analyse d'une frame
   * **Multiple Frames** : Analyse de plusieurs frames
   * **Trend Analysis** : Analyse des tendances

Analyse de tendances
~~~~~~~~~~~~~~~~~~~

L'analyse de tendances permet de :

* Suivre l'évolution du stock dans le temps
* Identifier les patterns de consommation
* Visualiser les tendances par produit
* Analyser frame par frame

6. Export des résultats
----------------------

Formats disponibles
~~~~~~~~~~~~~~~~~~

1. **JSON** : Données structurées pour l'intégration
2. **CSV** : Tableau pour l'analyse Excel
3. **Rapport texte** : Résumé détaillé lisible

Contenu des exports
~~~~~~~~~~~~~~~~~~

* Inventaire détaillé par produit
* Pourcentages de stock
* Recommandations de réapprovisionnement
* Timestamp d'analyse

API et intégration
==================

Structure des résultats JSON
----------------------------

.. code-block:: json

   {
     "summary": {
       "total_products_detected": 15,
       "estimated_missing_products": 3,
       "overall_stock_percentage": 83.3,
       "stock_levels": {
         "cocacola": {
           "current_count": 8,
           "missing_count": 2,
           "full_capacity": 10,
           "stock_percentage": 80.0
         }
       }
     }
   }

Pipeline programmatique
----------------------

.. code-block:: python

   from pipeline import EnhancedRetailPipeline

   # Initialisation
   pipeline = EnhancedRetailPipeline(
       yolo_model_path="models/individual_products.pt",
       cnn_model_path="models/classifier/best_lightweight_cnn.pth",
       void_model_path="models/void.pt",
       class_names=["cocacola", "oil", "water"],
       confidence_threshold=0.5
   )

   # Analyse
   results = pipeline.detect_and_classify_complete("image.jpg")

Optimisation et performances
============================

Recommandations
--------------

1. **Images** : Résolution optimale 1024x768 pixels
2. **Éclairage** : Éclairage uniforme recommandé
3. **Angle** : Vue frontale perpendiculaire au rayon
4. **Qualité** : Images nettes sans flou de mouvement

Limites
-------

* Temps de traitement : 2-10 secondes par image
* Formats supportés : JPG, PNG, BMP, MP4, AVI, MOV
* Taille maximale recommandée : 10 MB par fichier

Dépannage
=========

Problèmes courants
-----------------

**Pipeline non initialisé :**

.. code-block:: text

   ⚠️ Please configure and initialize the pipeline in the sidebar first.

**Solution :** Vérifiez les chemins des modèles et cliquez sur "Initialize Pipeline"

**Erreur de modèle :**

.. code-block:: text

   ❌ Error initializing pipeline: [Errno 2] No such file or directory

**Solution :** Vérifiez que les fichiers de modèles existent aux chemins spécifiés

**Erreur de classe :**

.. code-block:: text

   Class names mismatch

**Solution :** Vérifiez que les noms de classes correspondent à ceux du modèle

Logs et débogage
---------------

* Les erreurs s'affichent directement dans l'interface
* Vérifiez la console pour les détails techniques
* Consultez les chemins de fichiers dans la configuration

Support et maintenance
======================

Mise à jour des modèles
-----------------------

1. Remplacez les fichiers dans le dossier ``models/``
2. Mettez à jour ``model_info.json`` si nécessaire
3. Redémarrez l'application

Sauvegarde des configurations
----------------------------

Les configurations sont sauvegardées dans la session Streamlit et doivent être reconfigurées à chaque redémarrage.

Cas d'usage avancés
===================

Intégration e-commerce
---------------------

* Surveillance automatique des stocks
* Alertes de réapprovisionnement
* Optimisation des achats

Analyse retail
--------------

* Études de comportement consommateur
* Optimisation des planogrammes
* Contrôle qualité des rayons

Surveillance en temps réel
--------------------------

* Caméras de surveillance intégrées
* Analyses périodiques automatisées
* Tableaux de bord temps réel

Transfer learning pour le modèle de classification
--------------------------------------------------

- Il est possible d’ajouter de nouvelles classes de produits.
- Préparez vos images et leurs classes en utilisant une approche de clustering pour faciliter l’annotation.
- Effectuez un entraînement par transfert (Transfer Learning) sur ces nouvelles classes.
- Réutilisez ensuite le modèle pour la classification.
- Utilisez le notebook `notebookClassification.ipynb`.


.. note::
   Cette documentation couvre l'utilisation complète de l'application Retail Shelf Analysis. Pour des questions spécifiques ou des fonctionnalités avancées, consultez le code source ou me contactez .

Références
==========

* `Documentation Streamlit <https://docs.streamlit.io/>`_
* `OpenCV Documentation <https://docs.opencv.org/>`_
* `YOLO Documentation <https://github.com/ultralytics/yolov5>`_
* `PyTorch Documentation <https://pytorch.org/docs/>`_
