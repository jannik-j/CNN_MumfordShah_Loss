## Implementation zur Bachelorarbeit Bildsegmentierung durch Deep Learning mit U-Net und dem Mumford-Shah-Funktional
### Grundlage
Die hier verwendete Methode wurde von Boah Kim und Jong Chul Ye in [*Mumford-Shah Mumford-Shah Loss Functional for Image Segmentation with Deep Learning*](https://doi.org/10.1109/TIP.2019.2941265) vorgestellt. Diese Implementation basiert auf der von Kim und Ye zur Verfügung gestellten: [https://github.com/jongcye/CNN_MumfordShah_Loss]. 

### Datensätze
Diese Implementation wurde für das Training und die Auswertung von Netzwerken auf den folgenden Datensätzen verwendet:
* Datensatz der [LiTS-Challenge](http://www.lits-challenge.com) zur Segmentierung von Lebertumoren
* Datensatz zur Axon-Myelin-Segmentierung aus dem [Beispieldatensatz](https://axondeepseg.readthedocs.io/en/latest/documentation.html#example-dataset) von [AxonDeepSeg](https://github.com/neuropoly/axondeepseg)

Dateien mit Präfix `LiTS-` bzw. `Axon-` enthalten Methoden für die jeweiligen Datensätze.

### Training und Auswertung
Die Skripte zum Trainieren von Netzwerken sind `LiTS_train_unet.py` und `Axon_train_unet.py`. Sie werden über die entsprechenden Batch-Skripte im Ordner `scripts` mit den nötigen Kommandozeilenargumenten aufgerufen.

Analog sind `LiTS_test_unet.py` und `Axon_test_unet.py` für die Auswertung der Netzwerke zuständig. Auch sie werden über Batch-Skripte aufgerufen.

Die wichtigsten Kommandozeilenargumente sind:
* --dataroot: Verzeichnis, indem die Trainings- und Testdaten liegen.
* --gpu_ids: IDs der GPUs, die für die Berechnungen verwendet werden sollen (-1, falls keine vorliegen).
* --batchSize: Größe der Minibatches beim Training. Bei der Auswertung hier 1 verwenden.
* --semi_rate: Größe der Semirate zum Weglassen von Labeln. Bei der Auswertung hier 1 verwenden.
* --lr: Startwert der Lernrate.
* --lr_decay_iters: Anzahl der Epochen, nach denen die Lernrate halbiert wird.
* --niter: Gesamtanzahl der Epochen.
* --input_nc: Anzahl der Kanäle des Eingabebildes in das Netzwerk.
* --output_nc: Anzahl der Klassen, in die segmentiert wird.
* --save_epoch_freq: Anzahl der Epochen, nach denen das Netzwerk im Training zwischengespeichert wird.
* --segType: Art der Segmentierung. Beim LiTS-Datensatz gibt es die Optionen (liver, tumor, both), beim Datensatz zur Axon-Myelin-Segmentierung gibt es die Optionen (tem, sem).
* --lambda_A: Trainingsparameter Beta zur Gewichtung der Kostenfunktionen.
* --checkpoints_dir: Verzeichnis, in das die Netzwerke gespeichert werden.
* --which_epoch: Epoche des Zwischenspeicherstandes, der zur Auswertung geladen werden soll

Für eine ausführliche Erklärung aller Kommandozeilenargumente sei auf `options/base_options.py` und `options/train_options.py` verwiesen.
Die Bennung der Argumente entstammt der Implementation, auf der Kim und Ye ihre aufgebaut haben: [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix]


Diese Implementation basiert auf dem Repository von Jong Chul Ye:

Paper
===============
* Mumford–Shah Loss Functional for Image Segmentation With Deep Learning
  * Authors: Boah Kim and Jong Chul Ye
  * published in IEEE Transactions on Image Processing (TIP)

Implementation
===============
A PyTorch implementation of deep-learning-based segmentation based on original cycleGAN code.
[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix] 
(*Thanks for Jun-Yan Zhu and Taesung Park, and Tongzhou Wang.)

* Requirements
  * Python 2.7
  * PyTorch 1.1.0

Main
===============
* Training: LiTS_train_unet.py which is handled by scripts/LiTS_train_unet.sh
* A code for Mumford-Shah loss functional is in models/loss.py.
  * 'levelsetLoss' and 'gradientLoss2d' classes compose our Mumford-Shah loss function.
