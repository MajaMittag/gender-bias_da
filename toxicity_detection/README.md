# Unintended bias in toxicity detection

## Contents
* **archive**: Archive of old files. They might need minor revisions before running.
* **create_IPTTS**: Materials and code for generating the synthetic test set. Run the notebook to generate it. 
* **data**: 
    * overall AUCs for each model.
    * dataset splits (the original, CDS, data supplemented, and randomly supplemented dataset).
    * identity terms and lemmas for data supplementation.
    * the synthetic dataset and a pickled version with the model predictions.
* **embeddings**: The word2vec embeddings.
* **mitigation**: Two data augmentation mitigation approaches: counterfactual data substitution and data supplementation.
* **models**: Various model initializations.
* **plots**: Various plots created for the thesis.
* **bias_metrics.py**: Implementation of the bias metrics used in the bias analysis file.
* **hen_occurrences.ipynb**: Identify occurrences of the word "hen" in the original dataset.
* **toxicity_detection.ipynb**: Implementation of five toxicity detection classifiers trained on DKhate and variations thereof.
* **unintended_bias_analysis.ipynb**: Measure unintended bias in toxicity classifers.
* **utils.py**: Utility functions used in the main files (detection and bias analysis).
