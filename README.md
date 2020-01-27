# DDPG Reinforcement Learning für PEPPER Roboter
Ziel dieses Projektes ist es, den Pepper Roboter eine Schiene mit einem Ball in einer Dimension balancieren zu lassen.
Dafür gibt es eine blaue Unterlage mit rotem Ball. Ein Balltracker ermittelt das Delta (Abstand Mittelpunkt der blauen Fläche
und Ball) und gibt diesen in einem Thread zurück.

### BallTracker - Tracking des Balls über Kamera
Der Balltracker erkennt den Ball auf einfarbigem Hintergrund und liefert ein Delta zum Mittelpunkt des Untergrunds zurück.
### ddpg - Policy Algorithmus für das Lernen 
@ https://github.com/pemami4911/deep-rl

# Ablauf
Eine Webcam wird per USB an den Linux Computer angeschlossen (/dev/video1) und ein Pepper Roboter sollte sich im selben
Netzwerk befinden. Alle Einstellungen sind in der Settings.py vorzunehmen.

## pepper_ddpg_random_set_collector
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-GzZ9BT48mU/0.jpg)](https://www.youtube.com/watch?v=-GzZ9BT48mU)

Es werden dem Roboter Zufallszahlen gesendet um eine willkürliche Bewegung zu erzeugen. Dabei wird über die Kamera der aktuelle state
beobachtet und die Informationen über den Anfangs- und Folgezustand sowie erhaltener Reward in eine Textdatei im json Format geschrieben.
Dies ist die Grundlage für das Offline Lernen der gesammelten Daten in ein Modell.

## pepper_ddpg_offline_trainer
Die gesammelten Zufallsbewegungen werden jetzt wieder in zufälliger Reihenfolge iterativ in einen replayBuffer geschrieben und miteinander kombiniert.
Dadurch entsteht ein auf den gesammelten Daten trainiertes Modell. Hierbei wird je nach Einstellung alle 250 Episoden das aktuelle Modell in einem eigenen Ordner gespeichert.

## pepper_ddpg_model_runner
Das vorher trainierte Modell wird angewandt und erzeugt basierend auf dem aktuellen State (Motorstellung und Delta) eine Action
um in einen möglichst guten Folgezustand zu kommen. Dafür werden die verschiedenen Modelle durch iteriert und deren Resultate ausgegeben.

## pepper_ddpg_model_evaluator
Erstellt ein Modell zum Vorhersagen der Folgezustände und Rewards basierend auf den Trainingsdaten. Supervised Learning Modell.
Dient der Evaluierung des mit Reinforcement Learning erstellten Modells.

## Abhängigkeiten
`pip install -I tensorflow==1.3.0 tflearn==0.3.2 numpy==1.16.6 commentjson==0.8.2 qi==1.6.14`

