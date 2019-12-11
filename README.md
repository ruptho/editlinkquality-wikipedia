# On the Relation of Edit Behavior, Link Structure, and Article Quality on Wikipedia

This repository contains Python Code for experiments conducted during creation of the paper "On the Relation of Edit Behavior, Link Structure, and Article Quality on Wikipedia" by Thorsten Ruprechter, Tiago Santos, and Denis Helic, submitted to ComplexNetworks'19 [1].

The following files exist: 
- editbehavior.py
  - Computes relative label frequencies and edit label transition probabilities for already classified articles using revision features retrieved via the framework provided by Yang et al. [2]
- linkstructure.py
  - Postprocessing and example computation of link metrics for the WikiLinkGraphs dataset, generated using the framework by Consonni et al. [3]
- stats.py
  - Helper file for permutation tests to assess significance of results
- util.py
  - Helper for reading/writing files as well as other utility functions
- example.ipynb
  - Example iPython notebook on how components are supposed to be used, and in which order

---
[1] Complex Networks 2019. The 8th International Conference on
Complex Networks and their Applications. (https://www.complexnetworks.org/)

[2] Yang, Diyi, et al. "Identifying Semantic Edit Intentions from Revisions in Wikipedia." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017.

[3] Consonni, Cristian, et al. "WikiLinkGraphs: A complete, longitudinal and multi-language dataset of the Wikipedia link networks." Proceedings of the International AAAI Conference on Web and Social Media. Vol. 13. No. 01. 2019.
