from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict
import statistics
import math
import numpy as np
import json
from typing import Any


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
class Trader:
    def __init__(self):
        # Initialize trader state
        self.resin_fair_value = 10000  # Initial guess for Rainforest Resin
        self.recent_kelp_prices = []   # Store recent kelp prices to detect trends
        self.kelp_moving_avg_short = None  # Short-term moving average
        self.kelp_moving_avg_long = None   # Long-term moving average
        
    def run(self, state: TradingState):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        
        # Initialize the result dict with empty lists for all products
        result = {product: [] for product in state.order_depths.keys()}
        
        # Handle each product separately
        for product in state.order_depths:
            if product == 'RAINFOREST_RESIN':
                result[product] = self.trade_resin(product, state)
                
            elif product == 'KELP':
                pass
                #result[product] = self.trade_kelp2(product, state)
        
        # Serialize our state
        kelp_prices_str = ",".join([str(price) for price in self.recent_kelp_prices])
        trader_data = f"{self.resin_fair_value}|{kelp_prices_str}"
        
        # We're not using conversions in the tutorial
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def trade_resin(self, product: str, state: TradingState) -> List[Order]:
        """Market making strategy for Rainforest Resin"""
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        position_limit = 50
        
        # Reconstruct our state if available
        if state.traderData and "|" in state.traderData:
            parts = state.traderData.split("|")
            if len(parts) > 0:
                try:
                    self.resin_fair_value = float(parts[0])
                except:
                    pass
        #our strategy is to buy any orders when the ask price is less than 10000 and sell any orders when the bid is over 10000
        #the STD of Resin is 1.5, so we will place buy orders at 9998.5 and sell at 10001.5 for any remaining orders
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            self.resin_fair_value = (best_ask + best_bid) / 2
        # Get the best bid and ask available
        if order_depth.sell_orders:
            best_ask, best_ask_amount = min(order_depth.sell_orders.items())
            best_ask_amount = -best_ask_amount  # Convert to positive
        else:
            best_ask, best_ask_amount = float('inf'), 0
            
        if order_depth.buy_orders:
            best_bid, best_bid_amount = max(order_depth.buy_orders.items())
        else:
            best_bid, best_bid_amount = 0, 0
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position  # For short positions
        if (best_ask < 10000):
            #purchase
            trade_volume = max(best_ask_amount,buy_capacity) 
            if trade_volume > 0:
                orders.append(Order(product,best_ask,trade_volume))
                summary = f"LIMIT BUY {int(trade_volume)} {product} at {best_ask} "
                logger.print(summary)
        elif (best_bid > 10000):
            #sell
            trade_volume = max(best_bid,sell_capacity) 
            if trade_volume > 0:
                orders.append(Order(product,best_bid,-trade_volume))
                summary = f"LIMIT SELL {int(trade_volume)} {product} at {be
                                                                         st_bid} "
                logger.print(summary)
        #adjust by doing remaining trades with buy and sell orders 

        return orders

    def trade_kelp2(self,product:str, state: TradingState) -> List[Order]:
        orders =[]
        return orders

