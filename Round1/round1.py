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
    def __init__(self):
        # Initialize trader state
        self.resin_fair_value = 10000  # Initial guess for Rainforest Resin
        self.resin_spread = 1.5        # Std dev of Resin is approximately 1.5
        self.recent_kelp_prices = []   # Store recent kelp prices to detect trends
        self.kelp_moving_avg_short = None  # Short-term moving average
        self.kelp_moving_avg_long = None   # Long-term moving average
        
    def run(self, state: TradingState):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        
        # Initialize the result dict with empty lists for all products
        result = {product: [] for product in state.order_depths.keys()}
        
        # Reconstruct our state if available
        if state.traderData and "|" in state.traderData:
            parts = state.traderData.split("|")
            if len(parts) > 0:
                try:
                    self.resin_fair_value = float(parts[0])
                except:
                    pass
                    
                if len(parts) > 1 and parts[1]:
                    try:
                        self.recent_kelp_prices = [float(p) for p in parts[1].split(",") if p]
                    except:
                        pass
        
        # Handle each product separately
        for product in state.order_depths:
            if product == 'RAINFOREST_RESIN':
                result[product] = self.trade_resin(product, state)
                
            elif product == 'KELP':
                result[product] = self.trade_kelp(product, state)
        
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
        
        # Update fair value if we have both bid and ask
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            self.resin_fair_value = (best_ask + best_bid) / 2
            
        # Calculate remaining buy/sell capacity
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position  # For short positions
        
        # Process all profitable sell orders (price > 10000)
        if order_depth.buy_orders:
            # Sort buy orders by price in descending order to sell at the highest prices first
            sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in sorted_buy_prices:
                if price > 10000:  # Profitable to sell
                    volume = order_depth.buy_orders[price]
                    sell_volume = min(volume, sell_capacity)
                    
                    if sell_volume > 0:
                        orders.append(Order(product, price, -sell_volume))
                        sell_capacity -= sell_volume
                        logger.print(f"LIMIT SELL {sell_volume} {product} at {price}")
                        
                    if sell_capacity <= 0:
                        break
        
        # Process all profitable buy orders (price < 10000)
        if order_depth.sell_orders:
            # Sort sell orders by price in ascending order to buy at the lowest prices first
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sorted_sell_prices:
                if price < 10000:  # Profitable to buy
                    volume = -order_depth.sell_orders[price]  # Convert to positive
                    buy_volume = min(volume, buy_capacity)
                    
                    if buy_volume > 0:
                        orders.append(Order(product, price, buy_volume))
                        buy_capacity -= buy_volume
                        logger.print(f"LIMIT BUY {buy_volume} {product} at {price}")
                        
                    if buy_capacity <= 0:
                        break
        
        # Market making - place orders at fair price +/- spread
        market_making_buy_price = round(10000 - self.resin_spread)
        market_making_sell_price = round(10000 + self.resin_spread)
        
        # Only place market making orders if we have capacity left
        if buy_capacity > 0:
            orders.append(Order(product, market_making_buy_price, buy_capacity))
            logger.print(f"MM BUY {buy_capacity} {product} at {market_making_buy_price}")
            
        if sell_capacity > 0:
            orders.append(Order(product, market_making_sell_price, -sell_capacity))
            logger.print(f"MM SELL {sell_capacity} {product} at {market_making_sell_price}")
            
        return orders

    def trade_kelp(self, product: str, state: TradingState) -> List[Order]:
        """Strategy for Kelp - placeholder for now"""
        orders = []
        
        # Here you would implement your Kelp trading strategy
        # This is just a placeholder
        
        return orders