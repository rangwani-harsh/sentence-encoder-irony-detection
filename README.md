# irony-detection-allennlp-basic

This is a basic model for Irony Detection Subtask held at SemEval 2018. The model is based on the allennlp library.

The model is composed of three basic components :-
* Reader - Responsible for reading the dataset in .txt files in the dataset folder.
* Model - The model module defines the neural net we want to use for the model. It is currently a sequence to sequence encoder whoose output is fed to a feed forward network for classification.
* Predictor - This part is required for running the demo and evaluate procedure.

The **experiments** directory contains the config files which contains the hyperparameters and the model configuration.

To train the model run:

``` allennlp train  experiments/averagedencoder.json  -s directory_path_to_save_model --include-package irony_model```

Requirements:
* Allennlp==0.7.0

TODO
* Add New Models
* Update Documentation and Readme

