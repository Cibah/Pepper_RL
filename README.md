# DDPG Reinforcement Learning für PEPPER Roboter
Ziel dieses Projektes ist es, den Pepper Roboter eine Schiene mit einem Ball in einer Dimension balancieren zu lassen.
Dafür gibt es eine blaue Unterlage mit rotem Ball. Ein Balltracker ermittelt das Delta (Abstand Mittelpunkt der blauen Fläche
und Ball) und gibt diesen in einem Thread zurück.

### BallTracker - Tracking des Balls über Kamera
### ddpg - Policy Algorithmus für das Lernen @ https://github.com/pemami4911/deep-rl

#Ablauf
Eine Webcam wird per USB an den Linux Computer angeschlossen (/dev/video1) und ein Pepper Roboter sollte sich im selben
Netzwerk befinden.

#pepper_ddpg_random_set_collector
Es werden dem Roboter Zufallszahlen gesendet um eine willkürliche Bewegung zu erzeugen. Dabei wird über die Kamera der aktuelle state
beobachtet und die Informationen über den Anfangs- und Folgezustand sowie erhaltener Reward in eine Textdatei im json Format geschrieben.

Dies ist die Grundlage für das Offline Lernen der gesammelten Daten in ein Modell.

#pepper_ddpg_offline_trainer
Die gesammelten Zufallsbewegungen werden jetzt wieder in zufälliger Reihenfolge iterativ in einen replayBuffer geschrieben und miteinander kombiniert.
Dadurch entsteht ein auf den gesammelten Daten trainiertes Modell.

#pepper_ddpg_model_runner
Das vorher trainierte Modell wird angewandt und erzeugt basierend auf dem aktuellen State (Motorstellung und Delta) eine Action
um in einen möglichst guten Folgezustand zu kommen. 
