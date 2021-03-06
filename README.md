## Implementation zur Bachelorarbeit *Bildsegmentierung durch Deep Learning mit U-Net und dem Mumford-Shah-Funktional*
### Grundlage
Die hier verwendete Methode wurde von Boah Kim und Jong Chul Ye in [*Mumford-Shah Loss Functional for Image Segmentation with Deep Learning*](https://doi.org/10.1109/TIP.2019.2941265) vorgestellt. Diese Implementation basiert auf dem von Kim und Ye zur Verfügung gestellten Code: [https://github.com/jongcye/CNN_MumfordShah_Loss]. 

### Datensätze
Diese Implementation wurde für das Training und die Auswertung von Netzwerken auf den folgenden Datensätzen verwendet:
* Datensatz der [LiTS-Challenge](http://www.lits-challenge.com) zur Segmentierung von Lebertumoren
* Datensatz zur Axon-Myelin-Segmentierung aus dem [Beispieldatensatz](https://axondeepseg.readthedocs.io/en/latest/documentation.html#example-dataset) von [AxonDeepSeg](https://github.com/neuropoly/axondeepseg)

Dateien mit Präfix `LiTS-` bzw. `Axon-` enthalten Methoden für die jeweiligen Datensätze.

### Training und Auswertung
Die Skripte zum Trainieren von Netzwerken sind `LiTS_train_unet.py` und `Axon_train_unet.py`. Sie werden über die entsprechenden Batch-Skripte im Ordner `scripts` mit den nötigen Kommandozeilenargumenten aufgerufen.

Analog sind `LiTS_test_unet.py` und `Axon_test_unet.py` für die Auswertung der Netzwerke zuständig. Auch sie werden über Batch-Skripte aufgerufen.

Die wichtigsten Kommandozeilenargumente sind:
* `--dataroot`: Verzeichnis, indem die Trainings- und Testdaten liegen.
* `--gpu_ids`: IDs der GPUs, die für die Berechnungen verwendet werden sollen (-1, falls keine vorliegen).
* `--batchSize`: Größe der Minibatches beim Training. Bei der Auswertung hier 1 verwenden.
* `--semi_rate`: Größe der Semirate zum Weglassen von Labeln. Bei der Auswertung hier 1 verwenden.
* `--lr`: Startwert der Lernrate.
* `--lr_decay_iters`: Anzahl der Epochen, nach denen die Lernrate halbiert wird.
* `--niter`: Gesamtanzahl der Epochen.
* `--input_nc`: Anzahl der Kanäle des Eingabebildes in das Netzwerk.
* `--output_nc`: Anzahl der Klassen, in die segmentiert wird.
* `--save_epoch_freq`: Anzahl der Epochen, nach denen das Netzwerk im Training zwischengespeichert wird.
* `--segType`: Art der Segmentierung. Beim LiTS-Datensatz gibt es die Optionen (liver, tumor, both), beim Datensatz zur Axon-Myelin-Segmentierung gibt es die Optionen (tem, sem).
* `--lambda_A`: Trainingsparameter Beta zur Gewichtung der Kostenfunktionen.
* `--checkpoints_dir`: Verzeichnis, in das die Netzwerke gespeichert werden.
* `--which_epoch`: Epoche des Zwischenspeicherstandes, der zur Auswertung geladen werden soll.

Für eine ausführliche Erklärung aller Kommandozeilenargumente sei auf `options/base_options.py` und `options/train_options.py` verwiesen.
Die Benennung der Argumente entstammt der Implementation, auf der Kim und Ye die ihre aufgebaut haben: [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix]

Um Netzwerke ohne die Mumford-Shah-Kostenfunktion zu trainieren, muss in `models/unet_model.py` ein Codeblock modifiziert werden. Der geänderte Block ist in der Datei als Kommentar hinterlegt. Außerdem muss in diesem Fall die Semirate direkt in `LiTS_getDatabase_unet.py` hinterlegt werden, als Kommandozeilenargument ist 1 zu übergeben.

### Datenstruktur
Der Datensatz der LiTS-Challenge enhält pro CT-Scan je eine `.nii`-Datei für die Daten und für die Label. Die nötige Struktur für diese Implementation ist die Folgende:
In dem in `--dataroot` angegebenen Verzeichnis müssen die Ordner `train/` und `test/` enthalten sein. Sie entsprechen der Trainings- und der Testmenge. Für jeden CT-Scan, der in einer dieser Mengen enthalten ist, gibt es einen eigenen Ordner unterhalb von `train/` bzw. `test/`. Dieser enthält für jede *Schicht* des Scans, eine eigene `.mat`-Datei, die das Bild der Schicht als `data` und die Label der Schicht als `labels` enthält. Diese Dateien sind fortlaufend ab 1 nummeriert. Die Konversion der Daten lässt sich beispielsweise mit Matlab durchführen.

Ein Beispiel für diese Struktur:  
LiTS  
-train  
--42  
---1.mat  
...  
---125.mat
            

Beim Datensatz zur Axon-Myelin-Segmentierung müssen ebenfalls in dem in `--dataroot` angegebenen Verzeichnis die Ordner `train/` und `test/` existieren. Unterhalb dieser liegen jeweils die Ordner `tem/` und `sem/` für die beiden Methoden der Bilderzeugung. In der Arbeit wurden nur die Bilder des Transmissionselektronenmikroskops (TEM) verwendet. Die Bilder des Rasterelektronenmikroskops (SEM) wurden nicht verwendet. In diesen Verzeichnissen liegen dann die unveränderten Ordner, die die Daten enthalten.

Ein Beispiel für diese Struktur:  
Axon  
-train  
--tem  
---20160718_nyu_mouse_13_0001

### Benötigte Pakete
Diese Implemenation ist auf Python 3.7.7 lauffähig mit den folgenden Paketen:
* numpy
* scipy
* PyTorch
* Pillow
* matplotlib
* visdom
