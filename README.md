# Snuplasser
Dette prosjektet har som mål å identifisere og klassifisere snuplasser(vendingområder) fra kartdata. Ved å sammenligne snuplassens størrelse 
med forhåndsdefinerte kategorier, bestemmer systemet hvilken type kjøretøy området passer for. Informasjon fra tilhørende "label"-data brukes 
for å validere klassifiseringen.

## Innhold
1. [Modellen](#hva-koden-modeller)
2. [Kjøring](#hvordan-koden-kjøres)
3. [Definisjon](#definisjon)
4. [Struktur](#struktur)
5. [Avhengigheter](#avhengigheter)


## Hva koden modeller
- Prosjektet modellerer klassifisering og lokalisering av snuplasser i kartbaserte bilder.
- Modellens oppgave er todelt:
    - **20 meter** -> personbil
    - **25 meter** -> varebil
    - **30 meter** -> trailer/større transport
    - **35+ meter** ->  større militære kjøretøy

## Hvordan koden kjøres
- Koden kjøres lokalt på PC.
- Python brukes som hovedspråk.
- Det benyttes også **PyTorch** (nevralt nettverk/ bildeanalyse).
- En input-fil(antatt et bilde) brukes som input, og klassifisering returneres.

## Definisjon
 - Snuplassen:
 - Kordinat:
 - Bounding Box:
 - Label:

## Struktur

## Avhengigheter
* Python 3.10+
* PyTorch
* ...