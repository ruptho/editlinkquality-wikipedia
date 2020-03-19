# "Understanding and Predicting Edit War Articles on Wikipedia" and "On the Relation of Edit Behavior, Link Structure, and Article Quality on Wikipedia"
This repository contains Python Code for experiments conducted during creation of the papers "Understanding and Predicting Edit War Articles on Wikipedia" and "On the Relation of Edit Behavior, Link Structure, and Article Quality on Wikipedia" by Thorsten Ruprechter, Tiago Santos, and Denis Helic, submitted to ComplexNetworks'19 [1, 2]. Additionally, we prepared a dataset of 4,800 articles under https://zenodo.org/record/3716098.

The following files exist: 
- editbehavior.py
  - Computes relative label frequencies and edit label transition probabilities for already classified articles using revision features retrieved via the framework provided by Yang et al. [3]
- linkstructure.py
  - Postprocessing and example computation of link metrics for the WikiLinkGraphs dataset, generated using the framework by Consonni et al. [4]
- regression.py
  - Logistc Regression functions and configurations. Provides code for statistical control using statsmodels as well as logistic regression prediction experiments using scikit-learn
- stats.py
  - Helper file for permutation tests to assess significance of results
- util.py
  - Helper for reading/writing files as well as other utility functions
- example.ipynb
  - Example iPython notebook on how components are supposed to be used, and in which order
- pdf
  - Folder containing the poster presented at ComplexNetworks'19
  
---
[1] Ruprechter, Thorsten, Tiago Santos, and Denis Helic. "Understanding and Predicting Edit War Articles on Wikipedia." (Under Review)

[2] Ruprechter, Thorsten, Tiago Santos, and Denis Helic. "On the Relation of Edit Behavior, Link Structure, and Article Quality on Wikipedia." International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.

[3] Yang, Diyi, et al. "Identifying Semantic Edit Intentions from Revisions in Wikipedia." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017.

[4] Consonni, Cristian, et al. "WikiLinkGraphs: A complete, longitudinal and multi-language dataset of the Wikipedia link networks." Proceedings of the International AAAI Conference on Web and Social Media. Vol. 13. No. 01. 2019.
