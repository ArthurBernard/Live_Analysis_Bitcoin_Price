#! /usr/bin/env python3
import asyncio
import websockets
import json
import time

# Arthur BERNARD

""" This classes work only for ticker requests """

#============================================================================#

class WS_Bitfinex:
    """ Class to connect with the websocket API of bitfinex """
    def __init__(self, conn=None, STOP=None):
        self.data = 'nothing'
        self.timer = float(time.time())
        self.conn = conn
        self.STOP = STOP
    
    def request_WS(self, channel, uri='wss://api.bitfinex.com/ws', **kwargs):
        parameters = {'event': 'subscribe', 'channel':channel}
        for key, val in kwargs.items():
            parameters.update({key:val})
            if key == 'pair':
                self.pair = val
        return self.WS_bitfinex(parameters, uri)
        
    async def WS_bitfinex(self, parameters, uri='wss://api.bitfinex.com/ws'):
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(parameters))
            print("Bitfinex request: {}".format(parameters))
            
            first_resp = await websocket.recv()
            second_resp = await websocket.recv()
            print('Bitfinex first response {}'.format(first_resp))
            print('Bitfinex second response {}'.format(second_resp))
            
            try:
                async for message in websocket:
                    new = json.loads(message).copy()
                    if new[1] != 'hb':
                        self.data = self.sort_data_ticker(new)
                    if self.STOP:
                        break
            finally:
                websocket.close()
                print('Bitfinex close the websocket')
        
    def sort_data_ticker(self, data):
        """
        Order of the data: channelID, bid, bid_vol, 
        ask, ask_vol, daily_change, percentChange,
        last_price, volume, 24hrHigh, 24hrLow.
        """
        names = ['channel', 'bid', 'bid_vol', 'ask', 
                 'ask_vol', 'daily_change', 'perc_change', 
                 'close', 'vol', 'high', 'low']
        new_data = {names[i]: data[i] for i in range(len(names))}
        new_data['pair'] = self.pair
        return new_data

#============================================================================#

class WS_Poloniex:
    """ Class to connect with the websocket API of poloniex """
    def __init__(self, conn=None, STOP=None):
        self.data = 'nothing'
        self.timer = float(time.time())
        self.conn = conn
        self.STOP = STOP
    
    def request_WS(self, channel, uri='wss://api2.poloniex.com', **kwargs):
        parameters = {'command': 'subscribe', 'channel':channel}
        for key, val in kwargs.items():
            parameters.update({key:val})
        return self.WS_poloniex(parameters, uri)
        
    async def WS_poloniex(self, parameters, uri='wss://api2.poloniex.com'):
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(parameters))
            print("Poloniex request: {}".format(parameters))
            
            first_resp = await websocket.recv()
            print('Poloniex first response {}'.format(json.loads(first_resp)))
            try:
                async for message in websocket:
                    new = json.loads(message).copy()
                    try:
                        if new[2][0] == 121:
                            self.data = self.sort_data_ticker(new[2])
                    except IndexError:
                        print('IndexError')
                    if self.STOP:
                        break
            finally:
                websocket.close()
                print('Poloniex close the websocket')
        
    def sort_data_ticker(self, data):
        """
        Order of the data: currencyPair, last, lowestAsk, 
        highestBid, percentChange, baseVolume, quoteVolume, 
        isFrozen, 24hrHigh, 24hrLow.
        """
        names = ['pairId', 'close', 'ask', 'bid', 'perc_change',
                 'vol', 'qvol', 'froz', 'high', 'low']
        new_data = {names[i]: data[i] for i in range(len(names))}
        if new_data['pairId'] == 121:
            new_data['pair'] = 'BTC_USDT'
        return new_data

#============================================================================#

class WS_OKEx:
    """ Class to connect with the websocket API of OKEx """
    def __init__(self, conn=None, STOP=None):
        self.data = 'nothing'
        self.timer = float(time.time())
        self.conn = conn
        self.t = time.time()
        self.STOP = STOP
    
    def request_WS(self, channel='ok_sub_', kind='spot_', uri='wss://real.okex.com:10440/websocket/okexapi', **kwargs):
        channel += kind
        for key, val in kwargs.items():
            if key == 'pair':
                channel += val
                self.pair = val
            elif key == 'lenght':
                channel += '_'+val
                self.lenght = val
            elif key == 'method':
                channel += '_'+val
        parameters = {'event': 'addChannel', 'channel':channel}
        return self.WS_okex(parameters, uri)
        
    async def WS_okex(self, parameters, uri='wss://real.okex.com:10440/websocket/okexapi'):
        print('start')
        async with websockets.connect(uri) as websocket:
            print("OKEx request: {}".format(parameters))
            await websocket.send(json.dumps(parameters))
            
            first_resp = await websocket.recv()
            print('OKEx first response {}'.format(json.loads(first_resp)))
            try:
                async for message in websocket:
                    new = json.loads(message).copy()
                    if isinstance(new, list):
                        self.data = self.sort_data_ticker(new[0]['data'])
                    if time.time() - self.t >= 28:
                        await websocket.send(json.dumps({'event': 'ping'}))
                        self.t = time.time()
                    if self.STOP:
                    	break
            finally:
                websocket.close()
                print('OKEx close the websocket')
        
    def sort_data_ticker(self, data):
        """
        Order of the data: {binary, channel, 'data': 
        {high, vol, last, low, buy, change, sell, 
        dayLow, close, dayHigh, open, timestamp}
        }
        """
        return {'pair': self.pair, 'close': data['close'], 'ask': data['buy'], 
                'bid': data['sell'], 'perc_change': data['change'], 
                'vol': data['vol'], 'high': data['high'], 'low': data['low']}

#============================================================================#

class WS_Binance:
    """ Class to connect with the websocket API of binance """
    def __init__(self, conn=None, STOP=None):
        self.data = {}
        self.timer = float(time.time())
        self.conn = conn
        self.STOP = STOP
    
    def request_WS(self, method, pair='', uri='wss://stream.binance.com:9443/ws/'):
        uri += pair+method
        self.pair = pair
        return self.WS_binance(uri)
        
    async def WS_binance(self, uri='wss://stream.binance.com:9443/ws/'):
        async with websockets.connect(uri) as websocket:
            try:
                async for message in websocket:
                    new = json.loads(message).copy()
                    if isinstance(new, dict):
                        self.data = self.sort_data_ticker(new)
                    if self.STOP:
                    	break
            finally:
                websocket.close()
                print('Binance close the websocket')
        
    def sort_data_ticker(self, data):
        """
        Order of the data: currencyPair, last, lowestAsk, 
        highestBid, percentChange, baseVolume, quoteVolume, 
        isFrozen, 24hrHigh, 24hrLow.
        """
        return {'pair': data['s'], 'close': data['c'], 'ask': data['a'], 
                'bid': data['b'], 'perc_change': data['P'], 'change': data['p'],
                'vol': data['v'], 'high': data['h'], 'low': data['l'], 'qvol': data['q']}
