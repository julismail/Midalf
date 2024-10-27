# Midalf
Multimodal malware image and audio late fusion for malware detection
This is the repository of our paper "Midalf - Multimodal Malware Image and Audio Late Fusion for Malware Detection" We developed a malware detection system with multimodal. Midalf transforms malware binaries into image and audio representations. The image representations were trained with MalSSL, and the audio representations were trained with CNN. The output of both model were combined with late fusion.


We tested the system on two Downstream tasks:

Malware family classification with Malimg dataset, and
Malware benign classification with Debi dataset.
link to our paper:

Dataset

The Maldeb Dataset is an Image representation of Malware-benign dataset. The Dataset were compiled from various sources malware repositories: The Malware-Repo, TheZoo,Malware Bazar, Malware Database, TekDefense. Meanwhile benign samples were sourced from system application of Microsoft 10 and 11, as well as open source software repository such as Sourceforge, PortableFreeware, CNET, FileForum. The samples were validated by scanning them using Virustotal Malware scanning services. The Samples were pre-processed by transforming the malware binary into grayscale images following rules from Nataraj (2011).

Malimg Dataset is malware image dataset from Nataraj Paper: https://vision.ece.ucsb.edu/research/signal-processing-malware-analysis

Build with

pytorch https://pytorch.org/
lightly https://docs.lightly.ai/self-supervised-learning/index.html
Usage

File training.py for training the models
File testing.py for testing the models
File cross-validation for cross-validation testing with k=5
Folder Maldeb Dataset contains the collected malware-benign dataset
Folder SSL contains benchmark SSL models
Folder Audio contains code for pre-process malware for audio classification
Contributors

Hendrawan
Budi Rahardjo
Yasuo Musashi
Tutun Juhana
Debi Amalia Septiyani
Halimul Hakim Khairul
Dani Agung Prastiyo
Contact

https://twitter.com/Jul_Ismail
