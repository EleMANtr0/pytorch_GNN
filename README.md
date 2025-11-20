It is a project im currently working on using pytorch and pytorch_geometric. python version 3.12

All data processing, training models and getting results are in directory processing
Project consists of making dataset (creating graphs, assigning features to nodes and on graph level, putting raman spectras respectively.
This part was contributed and modified from https://github.com/gordondowns/cs224w-project.git), building model structures, training,
saving models, saving charts and saving predictions in ordered way.

Training was perfomed using many different loss functions and using combined loss functions according to results of each individual loss function and to some randomness :) still got great result

dataset is made balanced for each wavelength. there is also a file describing dataset versions - data/processed/version_description.txt

Project uses graph model PaiNN from schnetpack https://github.com/atomistic-machine-learning/schnetpack.git

I also realized too late that i might better use same metric for validating and comparing models - kl divergence, so this part was modified
but still a lot of results are already gained with different metrics, so there is a code evaluating all selected by hand models 
and also getting json with sorted predictions from worst to best. Then trere is a code to evaluate results and make some images with combined predictions of different models

final best model in ensemble of 2 other models