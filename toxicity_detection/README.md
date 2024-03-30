# Unintended bias in toxicity detection

**A brief description of each file and folder follows:**

* archive: Archive of old files. They might need minor revisions before running.
* create_IPTTS: Materials and code for generating the synthetic test set. Run the notebook to generate it. 
* data: 
    * overall AUCs for each model.
    * dataset splits (the original, CDS, data supplemented, and randomly supplemented dataset).
    * identity terms XX.
    * the synthetic dataset and a pickled version with the model predictions.
* embeddings: The word2vec embeddings.
* mitigation: XX.
* models: XX.
* plots: Various plots created for the thesis.

* bias_metrics.py: Implementation of the bias metrics used in the main files.
* hen_occurrences.ipynb: Identify occurrences of the word "hen" in the original dataset.
* toxicity_detection.ipynb: XX.
* unintended_bias_analysis.ipynb: XX.
* utils.py: Utility functions used in the main files.