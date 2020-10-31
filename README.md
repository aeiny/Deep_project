# Final project in Deep Learning CS236781

This project is a continuation of a previous effort to develop a new attention mechanism which is based on the information stored in both the temporal and spectral domain of a signal. It was shown to perform well on ECG signals, and this justifies the motivation to apply this mechanism to other types of signals such as voice, photos and more.

Please see project Report.pdf for more details on the project.

## Requirements:

Listed in the Requirements.txt file.

## Create the data:

To create the data, run the script CreateData.py: edit the function call in the bottom of the script, and then run the script.

## Train:

To train, use the train_script.py file. Tweak the following hyper-parameters to your choice:

Model structure:

* nonlinearities
* gets_spectral_input
* sta_enabled

Training parameters:

* Adam - if true uses Adam optimizer, otherwise SGD
* lr - learning rate
* n_epochs
* checkpoint_every - the model is saved to the checkpoint/ directory every this number of epochs.
* batch_size

Experiment's parameters:

* sig_name_train/test
* noise_name_train/test
* train_name/test_name - suffix for the training/validation/testing the data. For example to use 5cos(0.2x) with uniform noise of [0,8], we use the name 'data_signal__exp(cosX)__amp-5_freq-0.2__noise_uniform_low-0_high-8.npz' and the "train" prefix will be added automatically. The test name will be used both for validation and for testing (the test/validation files are different).

The script will show the training loss and validation loss in every epoch, and in the end will run the model on the test set and save the results on both the train and the test set in a file in results/ folder.
