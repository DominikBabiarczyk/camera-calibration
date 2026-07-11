# generate_dataset

Ten katalog zawiera narzędzie do generowania syntetycznych obrazów szachownicy.

## Co jest zapisane na stałe?

- `NUM_SQUARES_X = 9` — liczba kratek w poziomie
- `NUM_SQUARES_Y = 6` — liczba kratek w pionie
- `SQUARE_SIZE_PX = 48` — rozmiar jednej kratki w pikselach
- `IMAGE_WIDTH = 640`, `IMAGE_HEIGHT = 480` — rozmiar całego obrazu

## Użycie

Uruchom:

```bash
python generate_dataset/generate_chessboard.py
```

Wygenerowane obrazy pojawią się w katalogu `generate_dataset/boards/`.
