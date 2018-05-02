# Live analysis of Bitcoin price from Bitfinex, OKEx, Binance and Poloniex

***

## Description:
This is a study project where I used **asynchronous programmation** to download data and analyse it in the same time, and I used **cython** to speed-up the analysis. I also used my own econometric tools, they are surely not the best ones, but I kept control over them.

## Demo:
The plots and the analyse are update continuously as the following pictures.

<a href="https://imgflip.com/gif/29i7jb"><img src="https://i.imgflip.com/29i7jb.gif" title="made at imgflip.com"/></a>

## Contents:
- __*Price_Analysis.py*__ is the main script for analyse the raw price of bitcoin in US dollar.
- __*First_Difference_Analysis.py*__ is the main script for analyse the first difference of the price of bitcoin in US dollar.
- __*websocket_API_data.py*__ some classes to download ticker from websocket API.
- __*econometric_tools.py*__ contain some statistic, econometric or optimization tools.
- __*script_cython.pyx*__ are some cython loops faster than python code.
- __*setup.py*__ is a script to compile the cython code in C (use: 'python setup.py build_ext --inplace').