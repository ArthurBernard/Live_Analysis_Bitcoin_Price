# Live analysis of Bitcoin price from Bitfinex, OKEx, Binance and Poloniex

***

## Description:
  This is a class project where I used **asynchronous programmation** to download data and analyse it in the same time, and I used **cython** to speed-up the analysis. I also used my own econometric tools, they are surely not the best ones, but I kept control over them.
  
  When *Price_Analysis.py* or *Return_Analysis.py* is run a screen split into four appears. At the top left there is the graph of the price (or return) of Bitcoin in USD for each exchange, at the top right and at down left there are respectively drawing the **autocorrelogram** and the **kernel density function**. And at the down right there are some statistics and the estimation of the **best order ARMA(p,q)**. Analysis and graphs are **continuously updated** as shown in the GIF below.

## Short demo:

<a href="https://imgflip.com/gif/29i7jb"><img src="https://i.imgflip.com/29i7jb.gif" title="made at imgflip.com"/></a>

## Contents:
  - __*Price_Analysis.py*__ is the main script for analyse the raw price of bitcoin in US dollar.
  - __*Return_Analysis.py*__ is the main script for analyse the first difference of the price of bitcoin in US dollar.
  - __*websocket_API_data.py*__ some classes to download ticker data from websockets API.
  - __*econometric_tools.py*__ contain some statistic, econometric or optimization tools.
  - __*script_cython.pyx*__ are some cython loops faster than python code.
  - __*setup.py*__ is a script to compile the cython code in C.

## Installation:
  If you clone the scripts and try to run it, you will have to compile the cython code before. Go in the folder containing the scripts, and from the command line: 
> $ python setup.py build_ext --inplace
