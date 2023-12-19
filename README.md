# SingleWordProductionDutch

Scripts to work with the intracranial EEG data from [here](https://osf.io/nrgx6/) described in this [article](https://www.nature.com/articles/s41597-022-01542-9).

## Dependencies
The scripts require Python >= 3.6 and the following packages
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/scipylib/index.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/) 
* [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)

## Repository content
To recreate the experiments, run the following scripts.
### [Linear Regression Model]
* clone this repo:
```commandline
git clone https://github.com/neuralinterfacinglab/SingleWordProductionDutch.git
```
* __extract_features.py__: Reads in the iBIDS dataset and extracts features which are then saved to './features'

* __reconstruction_minimal.py__: Reconstructs the spectrogram from the neural features in a 10-fold cross-validation and synthesizes the audio using the Method described by Griffin and Lim.

* __viz_results.py__: Can then be used to plot the results figure from the paper.

* __reconstuctWave.py__: Synthesizes an audio waveform using the method described by Griffin-Lim

* __MelFilterBank.py__: Applies mel filter banks to spectrograms.

* __load_data.mat__: Example of how to load data using Matlab instead of Python

### [Dual-dualGAN Model]
* clone this repo:
```commandline
git clone https://github.com/paulili08/my_dual-dualGAN.git
```
* __extract_features.py__: Reads in the iBIDS dataset and extracts features with PCA which are then saved to './features'

* __pre_process.py__: Convert EEG signal into image representation and slice EEG image and spectrogram. Generate train/test set.

* __zmodel.py__: DualGAN.

* __model.py__: DualGAN between Domain A and Domain O.

* __model2.py__: DualGAN between Domain O and Domain B.

* __main.py__: Configuration for model.py

* __main2.py__: Configuration for model2.py

* __train.py__: Training instruction.

* __test.py__: Testing instruction.

* __evaluation.py__: PCC, SSIM, PSNR and Perceptual Loss evaluation.

* __MelFilterBank.py__: Applies mel filter banks to spectrograms.

* __load_data.mat__: Example of how to load data using Matlab instead of Python

* __ops.py__: Convolution, deconvolution, activate function and loss function for DualGAN.

* __domain_adaptation.py__: For cross subject validation. Replace pre_process.py after executing extract_features.py. Replace "eeg2audio" into "cross_sub" when execute train/test.py.