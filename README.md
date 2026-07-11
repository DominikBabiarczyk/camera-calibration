# camera_calibration

Projekt do eksperymentow z kalibracja kamery przy uzyciu klasycznych metod oraz sieci neuronowych.

## Przygotowanie srodowiska

Kroki instalacji (zalecane: PDM):

1. Wejdz do katalogu projektu:

```bash
cd /home/dbabiarczyk/studia/semestr10/camera_calibration
```

2. Sprawdz, czy masz PDM:

```bash
pdm --version
```

3. Zainstaluj wszystkie zaleznosci z `pyproject.toml` i `pdm.lock`:

```bash
pdm install
```

4. Uruchamiaj komendy przez PDM (zawsze w tym repo):

```bash
pdm run python -m src.networks_tool.train --help
```

5. Szybki test, czy srodowisko dziala poprawnie:

```bash
pdm run python -m src.networks_tool.train \
	--source-dir data/aug_camera_test_seq \
	--sequence \
	--sequence-length 5 \
	--model-name cnn_lstm_sequence \
	--epochs 1 \
	--batch-size 4 \
	--lr 1e-4
```

Alternatywa bez PDM (istniejace `.venv`):

```bash
cd /home/dbabiarczyk/studia/semestr10/camera_calibration
source .venv/bin/activate
python -m pip install torch torchvision opencv-python matplotlib numpy pyyaml tqdm
```

## Generowanie syntetycznych obrazow

Najpierw wygeneruj bazowa szachownice:

```bash
python generate_dataset/generate_chessboard.py
```

Plik zostanie zapisany jako:

```text
generate_dataset/boards/chessboard_000.png
```

Nastepnie wygeneruj sekwencyjny dataset treningowy z parametrami kamery:

```bash
python generate_dataset/creating_various_perspectives/augment_perspectives.py \
	--input generate_dataset/boards/chessboard_000.png \
	--outdir data/aug_camera_test_seq \
	--config generate_dataset/creating_various_perspectives/camera_calibration_config.yaml \
	--count 50 \
	--sequences 20 \
	--seed 42
```

Wynik ma strukture wymagana przez trening sekwencyjny:

```text
data/aug_camera_test_seq/
	sequence_000/
		camera_params.yaml
		aug_000.png
		aug_001.png
		...
	sequence_001/
		camera_params.yaml
		aug_000.png
		...
```

Do datasetu treningowego uzywaj `camera_calibration_config.yaml`, bo generuje obrazy z perspektywa, dystorsja kamery i etykietami `camera_params.yaml`. Plik `perspective_config.yaml` sluzy raczej do zwyklej augmentacji perspektywy obrazu i nie zapisuje etykiet kalibracyjnych.

## Uruchamianie treningu

Szybki test treningu na jednej epoce:

```bash
PYTHONPATH="$PWD" python -m src.networks_tool.train \
	--source-dir data/aug_camera_test_seq \
	--sequence \
	--sequence-length 5 \
	--model-name cnn_lstm_sequence \
	--epochs 1 \
	--batch-size 4 \
	--lr 1e-4
```

Pelny trening modelu sekwencyjnego:

```bash
PYTHONPATH="$PWD" python -m src.networks_tool.train \
	--source-dir data/aug_camera_test_seq \
	--sequence \
	--sequence-length 5 \
	--model-name cnn_lstm_sequence \
	--epochs 50 \
	--batch-size 8 \
	--lr 1e-4
```

Mozna tez uzyc gotowego skryptu:

```bash
./run_sequence_training.sh 5 cnn_lstm_sequence 50 8 1e-4
```

Najlepszy checkpoint jest zapisywany do:

```text
outputs/calibration_net/best_model.pth
```
