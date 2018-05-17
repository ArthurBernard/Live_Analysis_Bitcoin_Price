#! /usr/bin/env python
import asyncio
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

import websockets
import numpy as np

from websocket_API_data import WS_Bitfinex, WS_Poloniex, WS_Binance, WS_OKEx
from econometric_tools import EconometricTools, StatisticTools, OptimizationTools

# Arthur BERNARD

""" This script is the live analysis of Bitcoin price on Bitfinex, Poloniex, Binance and OKEx """

ST = StatisticTools()
ET = EconometricTools()

#====================================================================================#

class AnalyseData:
    """ This class gather data from the websockets, analyse and plot it in live """
    def __init__(self, objs={}, keys=['close'], STOP=300):
        """ Parametrization """
        self.STOP = STOP
        self.names = []
        self.obj = []
        self.nb = 0
        self.data = {}
        self.df = {}
        for name, obj in objs.items():
            self.obj.append(obj)
            self.names.append(name)
            self.nb += 1
            for key in keys:
                self.df[name, key] = np.array([[]], dtype=np.float64)
        self.objs = objs
        self.timer = time.time()
        self.t = 0
        self.keys = keys
        self.time_loop = 1
        self.l = [{}, {}, {}]
        self.h = [None, None, None, None, None, None, None, None]
        self.color = {'Poloniex': 'b-', 'Bitfinex': 'r-', 'Binance': 'y-', 'OKEx': 'g-'}
        
    def __aiter__(self):
        """ Iterator """
        return self
    
    async def __anext__(self):
        """ Asynchronous generator of data from the websockets """
        if self.t ==0:
            await asyncio.sleep(5)
            self.timer = time.time()-0.5
        self.t += 1
        for name, obj in self.objs.items():
            for key in self.keys:
                self.data[name, key] = float(obj.data[key])
                self.df[name, key] = np.append(
                    self.df[name, key], 
                    np.array([[np.float64(obj.data[key])]]), 
                    axis=1
                )
        return self.data

    async def looper(self, ax):
        """ Asynchronous loop to analyse and plot data in live """
        last = None
        wait = 0.9
        reg = time.time()
        self.mean_loop = 0
        self.xlim_min, self.xlim_max = 0, 60
        self.ylim_min, self.ylim_max = 10000, 0
        self.timer = time.time()
        async for data in self:
            self.t1 = time.time()
            if not last:
                last = data.copy()
            text = self.text(
                self.df['Poloniex', 'close'], 
                self.df['Bitfinex', 'close'], 
                self.df['Binance', 'close'], 
                self.df['OKEx', 'close'],
                Min=np.min, 
                Mean=ST.mean, 
                Max=np.max, 
                Var=ST.var
            )
            self.t3 = time.time()
            stat_txt.set_text(text)
            self.plot_price(data, last, [ax1])
            self.plot_histo(
                0, 
                [ax21, ax22, ax23, ax24, ax31, ax32, ax33, ax34], 
                [txt21, txt22, txt23, txt24, txt31, txt32, txt33, txt34]
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.0001)
            self.t4 = time.time()
            last = data.copy()
            self.time_loop = time.time() - self.timer
            self.mean_loop = (self.mean_loop*(self.t - 1) + self.time_loop)/self.t
            self.timer = time.time()
            if self.mean_loop > 1.01 and wait > 0:
                wait -= 0.02
                self.mean_loop -= 0.005
            await asyncio.sleep(wait)
            if self.t >= self.STOP:
                for gen in self.obj:
                    gen.STOP = True
                break

    def plot_price(self, data, last, ax=[]):
        """ Plot data """
        n = len(ax)
        rescale = None
        if self.t > self.xlim_max or self.t == 1:
            self.xlim_max *= 1.5
            rescale = True
        if min(data.values()) < self.ylim_min:
            self.ylim_min = min(data.values()) - 5
            rescale = True
        if max(data.values()) > self.ylim_max:
            self.ylim_max = max(data.values()) + 5
            rescale = True
        if rescale:
            self.width()
            for i in range(n):
                ax[i].clear()
                ax[i].set_ylim(self.ylim_min, self.ylim_max)
                ax[i].set_xlim(self.xlim_min, self.xlim_max)
                for name in self.names:
                    self.l[i][name], = ax[i].plot(
                        np.arange(self.t),
                        self.df[name, self.keys[i]].flatten(),
                        self.color[name], 
                        lineWidth=self.lwidth
                    )
                    ax[i].legend(self.names)
                    ax[i].set_title('Prices on several exchanges', fontsize=18, y=1.05)
                    ax[i].set_xlabel('t', x=1.1, fontsize=15)
                    ax[i].set_ylabel('BTCUSD', y=1, rotation=0, fontsize=14)
            rescale = False
        else:
            for i in range(n):
                for name in self.names:
                    self.l[i][name].set_data(
                        np.arange(self.t), 
                        self.df[name, self.keys[i]].flatten()
                    )
        self.t2 = time.time()

    def plot_histo(self, data, ax, txt):
        """ Plot the autocorrelation and the density of the data """
        point = [' ', '.', '..', '...']
        h = int(self.t**0.5)
        for i in range(len(ax)//2):
            if np.var(self.df[self.names[i], 'close']) > 1 and self.t > 2:
                if not self.h[i]:
                    ax[i].clear()
                    self.h[i] = ax[i].bar(
                        range(h+1), 
                        ET.auto_corr_emp_vect(self.df[self.names[i], 'close'].T, h=h).reshape(-1), 
                        width=0.9, 
                        color=self.color[self.names[i]][0]
                    )
                    if i == 0:
                        ax[i].set_title('Auto-correlogram', position=(1.15, 1.1), fontsize=18)
                        ax[i].tick_params(axis='x', labelbottom='off')
                    elif i == 1:
                        ax[i].tick_params(axis='both', labelbottom='off', labelleft='off')
                    elif i == 2:
                        ax[i].set_xlabel('Lag order', x=1.12, fontsize=14)
                        ax[i].set_ylabel(r'$\rho_k$', y=1, rotation=0, fontsize=17)
                    elif i == 3:
                        ax[i].tick_params(axis='y', labelleft='off')
                else:
                    gama = ET.auto_corr_emp_vect(self.df[self.names[i], 'close'].T, h=h)
                    for j in range(len(self.h[i])):
                        self.h[i][j].set_height([gama[j]])
                    if h > len(self.h[i]):
                        self.h[i] = None 
            else:
                txt[i].set_text('Loading{}'.format(point[int((self.t+16)%int((self.t+16)/4))]))
        for i in range(len(ax)//2, len(ax)):
            minus = np.min(self.df[self.names[i-4], 'close'])
            maxi = np.max(self.df[self.names[i-4], 'close'])
            if self.t > 2 and maxi - minus > 1:
                density = ET.kernel_density(self.df[self.names[i - 4], 'close'].T).flatten()
                grid = np.linspace(minus, maxi, self.t)
                if not self.h[i]:
                    ax[i].clear()
                    self.h[i], = ax[i].plot(grid, density, self.color[self.names[i - 4]], lineWidth=2)
                    if i == 6:
                        ax[i].set_xlabel('BTCUSD', x=1.1, fontsize=14)
                        ax[i].set_ylabel('Density', y=1, rotation=0, fontsize=14)
                    elif i == 4:
                        ax[i].tick_params(axis='x', labelbottom='off')
                    elif i == 5:
                        ax[i].tick_params(axis='both', labelbottom='off', labelleft='off')
                    elif i == 7:
                        ax[i].tick_params(axis='y', labelleft='off')
                else:
                    self.h[i].set_data(grid, density)
                    if int((self.t+i-3)**0.5) > h:
                        self.h[i] = None
            else:
                txt[i].set_text('Loading{}'.format(point[int((self.t+16)%int((self.t+16)/4))]))
        
    def text(self, *args, **kwargs):
        """ Return the analysis as a text (string) """
        text = '{:^14} | {:^13} | {:^14}\n'.format(
            str(self.t)+r'$^{th}$ observations', 
            str(self.nb)+' exchanges', 
            'Stop in '+str(self.STOP - self.t)+' obs'
        )
        text += '======================================================\n'
        text += '{:21} | {:21}\n'.format(
            'Current time loop: '+str(round(self.time_loop, 2)), 
            'Mean time per loop: '+str(round(self.mean_loop, 2))
        )
        text += '======================================================\n'
        text += 'Exchanges | {:^8} | {:^8} | {:^8} | {:^8}\n'.format(
            'Poloniex', 'Bitfinex', 'Binance', 'OKEx')
        text += '----------+----------+----------+----------+----------\n'
        for name_f, funct in kwargs.items():
            text += '{:10}'.format(name_f)
            for ex in args:
                text += '| {:8.2f} '.format(funct(ex))
            text += '\n'
        text += '----------+----------+----------+----------+----------\n'
        text += '{:10}'.format('Best ARMA')
        for ex in args:
            if self.t > 10:
                text += '| {:^8} '.format(str(ET.best_ARMA(ex.T)))
            else:
                text += '| {:^8} '.format('Loading')
        text += '\n'
        text += '======================================================\n'
        if False:#self.t > 1:
            text += 'Timer: 1st part {:.2f}s | 2nd part {:.2f}FPS | \n'.format(
                self.t1 - self.timer, 1/(self.t2 - self.t1)
            )
            text += 'total {:.2f}FPS | 4th part {:.2f} FPS\n'.format(
                1/(self.t2 - self.t3), 1/(self.timer - self.t4)
            )
        else:
            text += '\n{:^}\n'.format('Arthur BERNARD')
        return text

    def width(self):
        """ Actualize the linewidth to keep clean the plot """
        self.lwidth = int(((self.STOP - self.t)/self.STOP)*15)/10+0.3

#====================================================================================#
    
if __name__ == '__main__':
    polo = WS_Poloniex()
    bitf = WS_Bitfinex()
    bina = WS_Binance()
    okex = WS_OKEx()

    obj = {'Poloniex': polo, 'Bitfinex': bitf, 'Binance': bina, 'OKEx': okex}
    keys = ['close']
    STOP = 600

    gen = AnalyseData(obj, keys, STOP)

    fig = plt.figure('Analyze')
    ax1 = fig.add_subplot(2, 2, 1)
    ax23 = fig.add_subplot(4, 4, 7)
    ax21 = fig.add_subplot(4, 4, 3, sharex=ax23)
    ax24 = fig.add_subplot(4, 4, 8, sharey=ax23)
    ax22 = fig.add_subplot(4, 4, 4, sharey=ax21, sharex=ax24)

    ax33 = fig.add_subplot(4, 4, 13)
    ax31 = fig.add_subplot(4, 4, 9, sharex=ax33)
    ax34 = fig.add_subplot(4, 4, 14, sharey=ax33)
    ax32 = fig.add_subplot(4, 4, 10, sharey=ax31, sharex=ax34)
    ax4 = fig.add_subplot(2, 2, 4)
    ax = [ax1, ax21, ax31]

    txt21 = ax21.text(0.3, 0.5, 'Loading', family='monospace')
    txt22 = ax22.text(0.3, 0.5, 'Loading', family='monospace')
    txt23 = ax23.text(0.3, 0.5, 'Loading', family='monospace')
    txt24 = ax24.text(0.3, 0.5, 'Loading', family='monospace')
    txt31 = ax31.text(0.3, 0.5, 'Loading', family='monospace')
    txt32 = ax32.text(0.3, 0.5, 'Loading', family='monospace')
    txt33 = ax33.text(0.3, 0.5, 'Loading', family='monospace')
    txt34 = ax34.text(0.3, 0.5, 'Loading', family='monospace')
    for a in [ax21, ax22, ax23, ax24, ax31, ax32, ax33, ax34]:
        a.axis('off')
    txt = [txt21, txt22, txt23, txt24, txt31, txt32, txt33, txt34]
    stat_txt = ax4.text(0, -0.2, '', family='monospace')
    ax4.axis('off')

    name = ['Close']
    ax1.legend(['Poloniex', 'Bitfinex', 'Binance', 'OKEx'])
    ax1.set_title('Prices on several exchanges', fontsize=18, y=1)

    plt.ion()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    sns.despine()
    sns.despine(ax=ax21 , bottom=True)
    sns.despine(ax=ax31, bottom=True)
    sns.despine(ax=ax22, bottom=True, left=True)
    sns.despine(ax=ax32, bottom=True, left=True)
    sns.despine(ax=ax24, left=True)
    sns.despine(ax=ax34, left=True)
    sns.set(context='notebook', style='dark', palette='dark', color_codes=True)
    plt.show()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        polo.request_WS(1002),
        bitf.request_WS('ticker', pair='BTCUSD'),
        bina.request_WS(method='@ticker', pair='btcusdt'),
        okex.request_WS(channel='ok_sub_', kind='spot_', pair='btc_usdt', method='_ticker'),
        gen.looper(ax),
    ))
    print('End of the execution')