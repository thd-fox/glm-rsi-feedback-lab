# GLM Feedback Lab

Zwei feste GPU-Gruppen auf einem Server mit acht NVIDIA B300 wechseln sich beim Serving und beim Training von GLM-5.3-Flash ab. Jede Gruppe verfügt über vier Karten. Ein lokaler Router hält die Chat-API auf Port 8000 erreichbar, während die nächste Modellgeneration vorbereitet wird.

| Betriebsphase | Gruppe A · 4 B300 | Gruppe B · 4 B300 |
|---|---|---|
| Erste Runde | Live-Anfragen und Feedback-Korrekturen | Trainieren, exportieren und prüfen |
| Vergleich | Bisheriges Modell beantwortet Testfälle | Kandidat beantwortet dieselben Testfälle |
| Nach erfolgreicher Freigabe | Laufende Antworten beenden, nächste Trainingsrunde übernehmen | Live-Anfragen und Feedback-Korrekturen |
| Bei abgelehntem Kandidaten | Bleibt aktiv | Kandidat wird nicht freigegeben |

**Warum BF16?** Für das vollständige GLM-5.3-Flash dokumentiert NVIDIA zwei B300 in NVFP4 (ca. 205 GB Gewichte) und vier B300 in FP8 (ca. 330 GB) oder BF16 (ca. 645 GB). Alle Varianten enthalten das 320-Milliarden-Parameter-Modell. BF16 erlaubt es, verifizierte Adapter in die Ausgangsgewichte einzurechnen und beide Generationen in derselben Präzision zu vergleichen. NVIDIAs Profile betreffen Inferenz. Vier-Karten-QLoRA und selbst erzeugte Checkpoints brauchen einen eigenen Hardwaretest. [NVIDIA-Profilmatrix](https://docs.nvidia.com/nim/large-language-models/2.1.4/reference/support-matrix.html#glm-5-3-flash) · [Gewichtedownloads](https://docs.nvidia.com/nim/large-language-models/2.1.4/get-started/advanced/get-started-glm-5.3-flash.html) · [Z.ai](https://docs.z.ai/guides/vlm/glm-5.3-flash)

## Start auf dem Server

Vorausgesetzt werden Linux, Python ≥3.9, Docker mit NVIDIA Container Toolkit, der lokale [BF16-Checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) und die unten genannten Images. Für Export und Speichertest braucht der Host einen CUDA-Build von llama.cpp mit GLM5-Next-Unterstützung sowie die Python-Abhängigkeiten des Konverters. Die temporären BF16-Arbeitskopien benötigen je nach Dateisystem mehrere Terabyte Speicherplatz. [llama.cpp-Revision für den Lasttest](https://github.com/ggml-org/llama.cpp/tree/e50c254a22c590c9d2a60db86690d48d42497e2b)

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[checkpoint]'
docker pull nvcr.io/nim/nvidia/sglang-model-free-nim:2.1.4-variant-glm-5.3-flash
docker pull axolotlai/axolotl:main-latest
glm-rotate init --state runs/rotation --model /models/GLM-5.3-Flash-BF16
glm-route --state runs/rotation --port 8000
```

Den Router als eigenen Prozess laufen lassen. Das vorhandene Guardrail-Gateway kann seine Anfragen an `http://127.0.0.1:8000/v1` schicken. Der Modellalias lautet `glm-5.3-flash`. In einer zweiten Shell mit aktivierter Umgebung:

```sh
glm-feedback --cases examples/cases.jsonl --oracles examples/oracles.jsonl --out runs/feedback-01
glm-rotate round --state runs/rotation --data runs/feedback-01/train.jsonl --checks examples/checks.jsonl --llama-dir /opt/llama.cpp --steps 1
```

Die synthetischen Beispiele zeigen das Datenformat. Wenn das Ausgangsmodell beide Testaufgaben schon löst, kann der Kandidat dort keinen Zugewinn zeigen und die Gruppe bleibt aktiv. Für eine belastbare Freigabe werden eigene, unabhängige Fehler- und Kontrollfälle benötigt.

Der [Trainingsablauf](TRAINING.md) erklärt Rollenwechsel, Wiederanlauf und Größenlimit. Der Ein-Karten-Lasttest gilt für den quantisierten Text-Export eines Kandidaten. Live läuft das vollständige Modell auf vier B300.

**Teststand:** Lokal geprüft mit echten kleinen Gewichtstensoren, lokalen HTTP-Diensten und simulierten Docker-/GPU-Schnittstellen. Der vollständige GLM-Lauf auf acht B300 steht aus. Die Tests laufen mit `python -m unittest discover -s tests`.
