# Training, Freigabe und Rollenwechsel

## Aus negativem Feedback lernen

Vor dem Import Dialoge bereinigen, Trainingsnutzung freigeben und verwandte Sitzungen, Vorlagen und Beinahe-Duplikate derselben Partition zuordnen. [cases.jsonl](examples/cases.jsonl) enthält ursprüngliche Aufgabe, abgelehnte Antwort und Feedback. Ein unabhängiger [Sollwert](examples/oracles.jsonl) bestimmt, ob tatsächlich ein Fehler vorliegt. Eine vom Nutzer vorgeschlagene Korrektur hat Vorrang. Sonst erzeugt das aktive Modell bis zu drei Alternativen. Nur die ursprüngliche Aufgabe und eine verifizierte Lösung gelangen nach `train.jsonl`. Für andere Aufgaben als JSON-Ausgabe kann die Python-Bibliothek `prepare_sft` einen eigenen Prüfer entgegennehmen.

## Was eine Runde ausführt

1. Die aktive Vierergruppe bedient weiterhin den Router. Die andere bekommt den schreibgeschützten BF16-Elterncheckpoint und die freigegebenen SFT-Zeilen. Axolotl führt QLoRA mit FSDP2 in einem Offline-Container aus. Der Job protokolliert das verwendete Image, den Datenhash und die Gewichtsprüfsummen. Ein Trainingsschritt dient zur Hardwareprüfung. Für einen regulären Lauf `--steps` weglassen oder die gewünschte Schrittzahl angeben. Host-RAM beim Laden und Speichern gesondert beobachten. [Axolotl-Referenz für GLM](https://docs.axolotl.ai/docs/models/glm53-flash.html)
2. Der Adapter wird in seinen BF16-Elterncheckpoint integriert. Aus diesem Kandidaten entsteht ein zusammengeführter Q4_K_M-Text-Export. Auf einer physischen B300 des freien Viererblocks misst `glm-fit` das Laden und die Inferenz. Standardprofil sind ein Slot, 4.096 Tokens und 8 GiB im HBM belegte Reserve. Die Prüfung verlangt GPU-Platzierung aller Modellschichten und Speicherpuffer für Gewichte und Modellzustand. Der Bericht gilt für den gemessenen Kontext, Modellhash und Server-Build.
3. Die bisherige und die neue BF16-Instanz beantworten unabhängig dieselben zurückgehaltenen [Testfälle](examples/checks.jsonl). Der Test unterscheidet Zielaufgaben und Kontrollen. Wiederholte Trainingsprompts, ein fehlender Zielgewinn oder neu verursachte Fehler verhindern die Freigabe. Die integrierte Prüfung erfasst überprüfbare JSON-Aufgaben. Allgemeine Fähigkeiten und anwendungsspezifische Policy-Anforderungen brauchen eigene Tests. [Gepaarte Evaluation bei Fan et al.](https://arxiv.org/html/2611.21885v2#S3.SS6)
4. Nach Freigabe wechselt der Router auf die neue aktive Gruppe. Bestehende Streams dürfen zu Ende laufen. Neue Anfragen erreichen denselben lokalen Endpunkt. Die ehemalige Serving-Gruppe übernimmt in der nächsten Runde das Training.

Bei einer Unterbrechung `glm-rotate recover --state runs/rotation` verwenden. `runs/rotation/current.json` zeigt aktive Gruppe und Generation. Die Ports 8100/8101 sind gruppenspezifisch. Anwendungen verwenden den Router auf Port 8000.

## Größer werden

`--extra-width 256` bei `glm-rotate round` fügt den dichten SwiGLU-Schichten des Textdecoders je 256 Kanäle hinzu. Die neuen Ausgangsgewichte starten bei null. Damit bleibt die berechnete Ausgangsfunktion vor dem Training erhalten. Im folgenden Lauf können die neuen Kanäle über die Ausgabeprojektionen zum Ergebnis beitragen. Der Spielraum stammt aus dem Lasttest des Elternmodells. Jede gewachsene Generation benötigt einen eigenen erfolgreichen Lasttest.

Die Grenze von einer B300 bezieht sich auf die quantisierte Textversion einschließlich Laufzeitbedarf. Die BF16-Variante und der multimodale Vier-GPU-Server haben andere Speicheranforderungen. Ein Größen- oder Qualitätsgewinn belegt zunächst nur den jeweils geprüften Betriebsfall.

## Einzelne Werkzeuge

`glm-fleet plan` zeigt die GPU-Aufteilung für eigenständige Inferenz ohne gestartete Hardware. `glm-train --dry-run` und `glm-export --dry-run` zeigen den jeweiligen Aufruf mit lokalen Eingaben. `glm-checkpoint`, `glm-candidate`, `glm-fit` und `glm-evaluate` können separat ausgeführt werden. Die Werkzeuge für die Rotation koordinieren deren Reihenfolge und behalten den veröffentlichten Zustand bei abgelehnten Kandidaten bei.
