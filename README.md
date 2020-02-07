# Acted Facial Expressions In The Wild.
Continuous affect prediction in the wild is a very
interesting problem and is challenging as continuous prediction
involves heavy computation. We make an attempt
to predict continuous emotion dimensions i.e., valence and
arousal on extended Affect Wild Net (AffWid2) database.
AffWild2 database consists of videos in the wild labelled for
valence and arousal at frame level. It also consists of annotations
for seven discrete expressions such as neutral, anger, disgust,
fear, happiness, sadness and surprise. We used a bi-modal
approach by fusing audio and visual features and train a
sequence-to-sequence model that is based on Gated Recurrent
Units (GRU) and Long Short Term Memory (LSTM) network.
We show experimental results on validation data.

# Our experimental results are given below:
Baseline Model:  Valence: 0.14, Arousal: 0.24 <br />
Our Model (Audio+Video(ExpNet)+face(Openface)): Valence: 0.22, Arousal:0.34<br />


