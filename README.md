# MIDALF - Multimodal malware image and audio late fusion for malware detection

This is the repository of our paper *"MIDALF - Multimodal Malware Image and Audio Late Fusion for Malware Detection"*. We developed a malware detection system with multimodal. Midalf transforms malware binaries into image and audio representations. The image representations were trained with MalSSL, and the audio representations were trained with CNN. The output of both model were combined with late fusion. We tested the system on the Bodmas dataset and Maldeb dataset.

link to our paper:

**Dataset**

The Maldeb Dataset is an Image representation of Malware-benign dataset. The Dataset were compiled from various sources malware repositories: The Malware-Repo, TheZoo,Malware Bazar, Malware Database, TekDefense. Meanwhile benign samples were sourced from system application of Microsoft 10 and 11, as well as open source software repository such as Sourceforge, PortableFreeware, CNET, FileForum. The Maldeb dataset can be accessed at https://ieee-dataport.org/documents/maldeb-dataset

The samples were validated by scanning them using Virustotal Malware scanning services. The Samples were pre-processed by transforming the malware binary into grayscale images following rules from Nataraj (2011).

Malimg Dataset is malware image dataset from Nataraj Paper: https://vision.ece.ucsb.edu/research/signal-processing-malware-analysis

Bodmas dataset is introduced by Yang et.al, https://whyisyoung.github.io/BODMAS/

**Build with**

pytorch https://pytorch.org/
lightly https://docs.lightly.ai/self-supervised-learning/index.html

**Usage**

File multimedia.py for testing the model
File audio.py for training the model
The SSL classifier can be accessed at https://github.com/julismail/Self-Supervised
Folder GAN consist of GAN generated adversarial malware sample

**Contributors**

Hendrawan,
Budi Rahardjo,
Tutun Juhana,
Yasuo Musashi,
Dani Agung Prastiyo,
Dita Rananta.

**Contact**

https://twitter.com/Jul_Ismail
