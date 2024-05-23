from __future__ import annotations

from ._core import __doc__, __version__, wyswietlanie_sygnalu, generowanie_sin, generowanie_cos, generowanie_pros, generowanie_pilo, generowanie_kernela_gaussa_2D, generowanie_kernela_gaussa_1D
__all__ = ["__doc__", "__version__", "wyswietlanie_sygnalu", "generowanie_sin", "generowanie_cos", "generowanie_pros", "generowanie_pilo", "generowanie_kernela_gaussa_2D", "generowanie_kernela_gaussa_1D"]

import numpy as np
from scipy.io import wavfile
import imageio

amplituda = 1.0
okres = 1.0
samplowanie = 1000.0

def main():
    wybor = input("Wybierz co zrobic: (wizualizacja, generowanie, filtracja1D, filtracja2D): ")
    if wybor == "generowanie":
        generowanie()
    elif wybor == "wizualizacja":
        plik_we = input("Podaj nazwe pliku audio .wav: ")
        wizualizacja(plik_we)
    elif wybor == "filtracja1D":
        plik_we = input("Podaj nazwe pliku wejsciowego .wav: ")
        plik_wy = input("Podaj nazwe pliku wyjsciowego .wav: ")
        sigma = float(input("Podaj odchylenie standardowe filtra Gaussa: "))
        filtracja_gaussa_1d(plik_we, plik_wy, sigma)
    elif wybor == "filtracja2D":
        obraz_we = input("Podaj nazwe pliku wejsciowego .img: ")
        obraz_wy = input("Podaj nazwe pliku wyjsciowego .img: ")
        sigma = float(input("Podaj odchylenie standardowe filtra Gaussa: "))
        filtracja_gaussa_2d(obraz_we, obraz_wy, sigma)
    else:
        print("Podano bledna wartosc")

def wizualizacja(plik_we):
    rate, plik = wavfile.read(plik_we)
    if len(plik.shape) > 1:
        plik = plik[:, 0]
    czas = np.arange(500) / rate
    sygnal = plik[:500]
    wyswietlanie_sygnalu(czas.tolist(), sygnal.tolist(), "Pierwsze 500 sampli sygnalu")

def generowanie():
    wybor = input("Wybierz wykres: (sin, cos, prostokatny, piloksztaltny): ")
    if wybor in ["sin", "cos", "prostokatny", "piloksztaltny"]:
        czestotliwosc = float(input("Podaj czestotliwosc sygnalu (Hz): "))
    if wybor == "sin":
        czas, sygnal = generowanie_sin(czestotliwosc, amplituda, okres, samplowanie)
        wyswietlanie_sygnalu(czas, sygnal, "Sygnal Sinusoidalny")
    elif wybor == "cos":
        czas, sygnal = generowanie_cos(czestotliwosc, amplituda, okres, samplowanie)
        wyswietlanie_sygnalu(czas, sygnal, "Sygnal Cosinusoidalny")
    elif wybor == "prostokatny":
        czas, sygnal = generowanie_pros(czestotliwosc, amplituda, okres, samplowanie)
        wyswietlanie_sygnalu(czas, sygnal, "Sygnal Prostokatny")
    elif wybor == "piloksztaltny":
        czas, sygnal = generowanie_pilo(czestotliwosc, amplituda, okres, samplowanie)
        wyswietlanie_sygnalu(czas, sygnal, "Sygnal Piloksztaltny")
    else:
        print("Podano bledna wartosc")

def filtracja_gaussa_1d(plik_we, plik_wy, sigma):
    rate, plik = wavfile.read(plik_we)
    if len(plik.shape) > 1:
        plik = plik[:, 0]
    kernel = np.array(generowanie_kernela_gaussa_1D(sigma))
    rozmiar_kernela = len(kernel)
    promien_kernela = rozmiar_kernela // 2
    przefiltrowany_plik = np.zeros_like(plik, dtype=float)
    padded_plik = np.pad(plik, (promien_kernela, promien_kernela), mode='reflect')
    for i in range(len(plik)):
        przefiltrowany_plik[i] = np.sum(padded_plik[i:i + rozmiar_kernela] * kernel)
    przefiltrowany_plik = np.asarray(przefiltrowany_plik, dtype=np.int16)
    wavfile.write(plik_wy, rate, przefiltrowany_plik)

def filtracja_gaussa_2d(obraz_we, obraz_wy, sigma):
    obraz = imageio.imread(obraz_we)
    kernel = np.array(generowanie_kernela_gaussa_2D(sigma))
    wysokosc_kernela, szerokosc_kernela = kernel.shape
    if obraz.ndim == 2:
        wysokosc, szerokosc = obraz.shape
        kanaly = 1
        obraz = obraz[:, :, None]
    else:
        wysokosc, szerokosc, kanaly = obraz.shape
    przefiltrowany_obraz = np.zeros_like(obraz)
    wysokosc_pad = wysokosc_kernela // 2
    szerokosc_pad = szerokosc_kernela // 2
    padded_obraz = np.pad(obraz, ((wysokosc_pad, wysokosc_pad), (szerokosc_pad, szerokosc_pad), (0, 0)), mode='reflect')
    for c in range(kanaly):
        for i in range(wysokosc):
            for j in range(szerokosc):
                region = padded_obraz[i:i + wysokosc_kernela, j:j + szerokosc_kernela, c]
                przefiltrowany_obraz[i, j, c] = np.sum(region * kernel)
    if kanaly == 1:
        przefiltrowany_obraz = przefiltrowany_obraz[:, :, 0]
    imageio.imwrite(obraz_wy, przefiltrowany_obraz.astype(np.uint8))
    
    
if __name__ == "__main__":
    main()