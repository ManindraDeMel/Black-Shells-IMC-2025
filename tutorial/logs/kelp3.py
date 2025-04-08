from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math
import numpy as np

class Trader:
    def __init__(self):
        # Initialize trader state
        self.resin_fair_value = 10000  # Initial guess for Rainforest Resin
        self.recent_kelp_prices = []   # Store recent kelp prices to detect trends
        self.kelp_moving_avg_short = None  # Short-term moving average
        self.kelp_moving_avg_long = None   # Long-term moving average
        
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        # Initialize the result dict with empty lists for all products
        result = {product: [] for product in state.order_depths.keys()}
        
        # Handle each product separately
        for product in state.order_depths:
            if product == 'RAINFOREST_RESIN':
                #result[product] = self.trade_resin(product, state)
                pass
            elif product == 'KELP':
                result[product] = self.trade_kelp2(product, state)
        
        # Serialize our state
        kelp_prices_str = ",".join([str(price) for price in self.recent_kelp_prices])
        trader_data = f"{self.resin_fair_value}|{kelp_prices_str}"
        
        # We're not using conversions in the tutorial
        conversions = 0
        
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
        
        # Calculate fair value based on the mid-price of the order book
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
            
        # Market making logic - buy low, sell high
        # Calculate acceptable prices with a margin
        margin = 1  # Adjust based on market conditions
        buy_price = self.resin_fair_value - margin
        sell_price = self.resin_fair_value + margin
        
        # Adjust based on our current position
        position_factor = 0.2 * (position / position_limit)
        buy_price -= position_factor * margin
        sell_price += position_factor * margin
        
        # Limit orders to maintain position limits
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position  # For short positions
        
        # Place buy order if the best ask is below our buy price
        if best_ask < buy_price and buy_capacity > 0:
            buy_volume = min(best_ask_amount, buy_capacity)
            if buy_volume > 0:
                orders.append(Order(product, best_ask, buy_volume))
                print(f"BUY {product}: {buy_volume}x at {best_ask}")
        
        # Place sell order if the best bid is above our sell price
        if best_bid > sell_price and sell_capacity > 0:
            sell_volume = min(best_bid_amount, sell_capacity)
            if sell_volume > 0:
                orders.append(Order(product, best_bid, -sell_volume))
                print(f"SELL {product}: {sell_volume}x at {best_bid}")
                
        # Optional: place limit orders at our prices
        if buy_capacity > 10:
            orders.append(Order(product, int(buy_price), 10))
            print(f"LIMIT BUY {product}: 10x at {int(buy_price)}")
            
        if sell_capacity > 10:
            orders.append(Order(product, int(sell_price), -10))
            print(f"LIMIT SELL {product}: 10x at {int(sell_price)}")
        
        return orders
    
    def trade_kelp2(self,product:str, state: TradingState) -> List[Order]:
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        position_limit = 50
        window_size = 20
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            self.recent_kelp_prices.append(mid_price)
            self.recent_kelp_prices = self.recent_kelp_prices[-1*window_size:]  # Keep only latest history

        if(len(self.recent_kelp_prices) >= window_size):
            average_kelp_price = statistics.mean(self.recent_kelp_prices)
            best_ask, best_ask_amount = min(order_depth.sell_orders.items())
            best_bid, best_bid_amount = min(order_depth.buy_orders.items())
            best_ask_amount = -best_ask_amount
        
            if(best_ask < average_kelp_price): # Buy
                trade_size = 0.5*(position_limit - position)
                trade_size = math.floor(min(trade_size, best_ask_amount))
                if trade_size > 0:
                            orders.append(Order(product, best_ask, trade_size))
                            print(f" BUY {product}: {trade_size}x at {best_ask}")
                        
            elif (best_bid > average_kelp_price): #hit the bid/sell
                trade_size = 0.5*(position_limit + position)
                trade_size = math.floor(min(trade_size, best_bid_amount))
                if trade_size > 0:
                            orders.append(Order(product, best_bid, -trade_size))
                            print(f"Sell {product}: {trade_size}x at {best_bid}")
            else: #market make
                trade_size = 0.25*(position_limit - position)
                orders.append(Order(product, average_kelp_price*0.85, trade_size))
                trade_size = 0.25*(position_limit + position)
                orders.append(Order(product, average_kelp_price*1.15, trade_size))
        return orders