from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import statistics
import math
import numpy as np
import json


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
        self.resin_spread = 1.5        # Initial spread for Resin
        self.resin_prices = []         # Store recent trades for volatility calculation
        self.resin_trade_volumes = []  # Track recent trade volumes
        
        # For KELP product
        self.kelp_prices = []
        self.kelp_fair_value = None
        
        # Maximum history to keep
        self.max_history = 100
        
    def run(self, state: TradingState):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        
        # Initialize the result dict with empty lists for all products
        result = {product: [] for product in state.order_depths.keys()}
        
        # Deserialize our state
        self.deserialize_state(state.traderData)
        
        # Handle each product separately
        for product in state.order_depths:
            if product == 'RAINFOREST_RESIN':
                result[product] = self.trade_resin(product, state)
            elif product == 'KELP':
                result[product] = self.trade_kelp(product, state)
        
        # Serialize our state
        trader_data = self.serialize_state()
        
        # We're not using conversions in this implementation
        conversions = 0
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def deserialize_state(self, trader_data: str):
        """Deserialize the trader state from the trader_data string"""
        if not trader_data:
            return
            
        try:
            parts = trader_data.split("|")
            
            # Restore resin fair value
            if len(parts) > 0 and parts[0]:
                self.resin_fair_value = float(parts[0])
                
            # Restore resin spread
            if len(parts) > 1 and parts[1]:
                self.resin_spread = float(parts[1])
                
            # Restore resin prices
            if len(parts) > 2 and parts[2]:
                price_strs = parts[2].split(",")
                self.resin_prices = [float(p) for p in price_strs if p]
                
            # Restore resin volumes
            if len(parts) > 3 and parts[3]:
                volume_strs = parts[3].split(",")
                self.resin_trade_volumes = [float(v) for v in volume_strs if v]
                
            # Restore kelp prices
            if len(parts) > 4 and parts[4]:
                kelp_price_strs = parts[4].split(",")
                self.kelp_prices = [float(p) for p in kelp_price_strs if p]
                
            # Restore kelp fair value
            if len(parts) > 5 and parts[5]:
                self.kelp_fair_value = float(parts[5])
        except Exception as e:
            logger.print(f"Error deserializing state: {e}")
    
    def serialize_state(self) -> str:
        """Serialize the trader state into a string"""
        resin_prices_str = ",".join([str(price) for price in self.resin_prices])
        resin_volumes_str = ",".join([str(vol) for vol in self.resin_trade_volumes])
        kelp_prices_str = ",".join([str(price) for price in self.kelp_prices])
        kelp_fair_value_str = str(self.kelp_fair_value) if self.kelp_fair_value else ""
        
        return f"{self.resin_fair_value}|{self.resin_spread}|{resin_prices_str}|{resin_volumes_str}|{kelp_prices_str}|{kelp_fair_value_str}"

    def trade_resin(self, product: str, state: TradingState) -> List[Order]:
        """Enhanced market making strategy for Rainforest Resin"""
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        position_limit = 50
        
        # Update trade history with recent market trades
        self.update_trade_history(product, state)
        
        # Calculate dynamic spread based on recent volatility
        self.calculate_dynamic_spread(product)
        
        # Get best bid and ask if available
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        
        # Update fair value based on market data
        self.update_fair_value(product, order_depth, position, best_bid, best_ask)
        
        # Calculate remaining buy/sell capacity
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        
        # Apply order book imbalance and position-based adjustments
        adjusted_fair_value = self.calculate_adjusted_fair_value(product, order_depth, position, position_limit)
        logger.print(f"Fair value: {self.resin_fair_value}, Adjusted: {adjusted_fair_value}, Spread: {self.resin_spread}")
        
        # Profit-taking logic for existing inventory
        orders.extend(self.profit_taking_orders(product, position, best_bid, best_ask, buy_capacity, sell_capacity))
        
        # Update capacities after profit taking
        buy_capacity = position_limit - (position + sum(order.quantity for order in orders if order.quantity > 0))
        sell_capacity = position_limit + (position - sum(-order.quantity for order in orders if order.quantity < 0))
        
        # Process all profitable opportunities
        buy_orders, updated_buy_capacity = self.process_buy_opportunities(
            product, order_depth, adjusted_fair_value, buy_capacity
        )
        orders.extend(buy_orders)
        buy_capacity = updated_buy_capacity
        
        sell_orders, updated_sell_capacity = self.process_sell_opportunities(
            product, order_depth, adjusted_fair_value, sell_capacity
        )
        orders.extend(sell_orders)
        sell_capacity = updated_sell_capacity
        
        # Apply inventory skew for market making
        skewed_orders = self.apply_inventory_skew(
            product, position, position_limit, adjusted_fair_value, buy_capacity, sell_capacity
        )
        orders.extend(skewed_orders)
        
        return orders
    
    def update_trade_history(self, product: str, state: TradingState):
        """Update trade history with recent market and own trades"""
        # Add market trades to history
        if product in state.market_trades:
            for trade in state.market_trades[product]:
                self.resin_prices.append(trade.price)
                self.resin_trade_volumes.append(trade.quantity)
                
        # Add own trades to history
        if product in state.own_trades:
            for trade in state.own_trades[product]:
                self.resin_prices.append(trade.price)
                self.resin_trade_volumes.append(trade.quantity)
                
        # Limit history size
        if len(self.resin_prices) > self.max_history:
            self.resin_prices = self.resin_prices[-self.max_history:]
            
        if len(self.resin_trade_volumes) > self.max_history:
            self.resin_trade_volumes = self.resin_trade_volumes[-self.max_history:]
    
    def calculate_dynamic_spread(self, product: str):
        """Calculate spread based on recent price volatility"""
        if len(self.resin_prices) >= 10:
            # Calculate standard deviation of recent prices
            std_dev = np.std(self.resin_prices[-30:]) if len(self.resin_prices) >= 30 else np.std(self.resin_prices)
            
            # Adjust spread based on volatility, but keep within reasonable bounds
            self.resin_spread = max(2.0, min(5.0, std_dev))
            logger.print(f"Dynamic spread calculated: {self.resin_spread} (std: {std_dev})")
        else:
            # Default spread if not enough data
            self.resin_spread = 1.5
    
    def update_fair_value(self, product: str, order_depth: OrderDepth, position: int, best_bid: int, best_ask: float):
        """Update fair value based on market data"""
        # Update based on mid-price if both bid and ask available
        if best_bid > 0 and best_ask < float('inf'):
            mid_price = (best_bid + best_ask) / 2
            # Gradual adjustment (80% old value, 20% new value) for stability
            self.resin_fair_value = 0.8 * self.resin_fair_value + 0.2 * mid_price
        
        # Mean reversion component - gradually pull towards 10000
        mean_reversion_factor = 0.5  # 5% adjustment towards mean per iteration
        self.resin_fair_value = (1 - mean_reversion_factor) * self.resin_fair_value + mean_reversion_factor * 10000
    
    def calculate_adjusted_fair_value(self, product: str, order_depth: OrderDepth, position: int, position_limit: int):
        """Calculate adjusted fair value based on order book imbalance and position"""
        adjusted_fair_value = self.resin_fair_value
        
        # Order book imbalance adjustment
        if order_depth.buy_orders and order_depth.sell_orders:
            buy_volume = sum(order_depth.buy_orders.values())
            sell_volume = sum(-v for v in order_depth.sell_orders.values())
            
            if sell_volume > 0:
                imbalance = buy_volume / sell_volume
                # Cap the imbalance effect
                imbalance_adjustment = min(2.0, max(-2.0, 0.01 * (imbalance - 1)))
                adjusted_fair_value += imbalance_adjustment
                logger.print(f"Imbalance: {imbalance:.2f}, Adjustment: {imbalance_adjustment:.2f}")
        
        # Position-based adjustment (encourage mean reversion)
        position_ratio = position / position_limit
        position_adjustment = -0.01 * position_ratio * self.resin_spread
        adjusted_fair_value += position_adjustment
        logger.print(f"Position: {position}, Adjustment: {position_adjustment:.2f}")
        
        return adjusted_fair_value
    
    def profit_taking_orders(self, product: str, position: int, best_bid: int, best_ask: float, 
                           buy_capacity: int, sell_capacity: int) -> List[Order]:
        """Generate profit-taking orders for existing inventory"""
        orders = []
        
        # Take profit on long positions when price is above fair value
        if position > 10 and best_bid > self.resin_fair_value + self.resin_spread:
            profit_taking_size = min(position, sell_capacity)
            if profit_taking_size > 0:
                orders.append(Order(product, best_bid, -profit_taking_size))
                logger.print(f"PROFIT TAKE (LONG): SELL {profit_taking_size} {product} at {best_bid}")
        
        # Take profit on short positions when price is below fair value
        if position < -10 and best_ask < self.resin_fair_value - self.resin_spread:
            profit_taking_size = min(-position, buy_capacity)
            if profit_taking_size > 0:
                orders.append(Order(product, best_ask, profit_taking_size))
                logger.print(f"PROFIT TAKE (SHORT): BUY {profit_taking_size} {product} at {best_ask}")
        
        return orders
    
    def process_buy_opportunities(self, product: str, order_depth: OrderDepth, 
                                fair_value: float, buy_capacity: int) -> (List[Order], int):
        """Process all buy opportunities (price < fair value)"""
        orders = []
        remaining_capacity = buy_capacity
        
        if order_depth.sell_orders and remaining_capacity > 0:
            # Sort sell orders by price in ascending order to buy at the lowest prices first
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sorted_sell_prices:
                # Only buy if price is below our fair value minus a small buffer
                if price < fair_value - 0.2:
                    volume = -order_depth.sell_orders[price]  # Convert to positive
                    buy_volume = min(volume, remaining_capacity)
                    
                    if buy_volume > 0:
                        orders.append(Order(product, price, buy_volume))
                        remaining_capacity -= buy_volume
                        logger.print(f"BUY OPPORTUNITY: {buy_volume} {product} at {price}")
                    
                    if remaining_capacity <= 0:
                        break
        
        return orders, remaining_capacity
    
    def process_sell_opportunities(self, product: str, order_depth: OrderDepth, 
                                 fair_value: float, sell_capacity: int) -> (List[Order], int):
        """Process all sell opportunities (price > fair value)"""
        orders = []
        remaining_capacity = sell_capacity
        
        if order_depth.buy_orders and remaining_capacity > 0:
            # Sort buy orders by price in descending order to sell at the highest prices first
            sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in sorted_buy_prices:
                # Only sell if price is above our fair value plus a small buffer
                if price > fair_value + 0.2:
                    volume = order_depth.buy_orders[price]
                    sell_volume = min(volume, remaining_capacity)
                    
                    if sell_volume > 0:
                        orders.append(Order(product, price, -sell_volume))
                        remaining_capacity -= sell_volume
                        logger.print(f"SELL OPPORTUNITY: {sell_volume} {product} at {price}")
                    
                    if remaining_capacity <= 0:
                        break
        
        return orders, remaining_capacity
    
    def apply_inventory_skew(self, product: str, position: int, position_limit: int, 
                           fair_value: float, buy_capacity: int, sell_capacity: int) -> List[Order]:
        """Apply inventory skew to market making orders"""
        orders = []
        
        # Calculate skew factor (between 0.3 and 0.7)
        skew = 0.5 + (position / (2 * position_limit))
        skew = max(0.45, min(0.55, skew))
        logger.print(f"Position: {position}, Skew factor: {skew:.2f}")
        
        # Apply skew to order sizes
        buy_size = int(buy_capacity * (1 - skew))
        sell_size = int(sell_capacity * skew)
        
        # Place layered orders (3 levels with decreasing size)
        levels = 3
        for i in range(levels):
            # Increase spread for each level away from fair value
            level_spread = self.resin_spread * (1 + 0.5 * i)
            
            # Calculate size for this level (decreasing size)
            level_buy_size = buy_size // (2**i) if i < levels - 1 else buy_size
            level_sell_size = sell_size // (2**i) if i < levels - 1 else sell_size
            
            # Update remaining sizes
            buy_size -= level_buy_size
            sell_size -= level_sell_size
            
            # Create buy order at this level
            if level_buy_size > 0:
                buy_price = round(fair_value - level_spread)
                orders.append(Order(product, buy_price, level_buy_size))
                logger.print(f"MM BUY L{i+1}: {level_buy_size} {product} at {buy_price}")
            
            # Create sell order at this level
            if level_sell_size > 0:
                sell_price = round(fair_value + level_spread)
                orders.append(Order(product, sell_price, -level_sell_size))
                logger.print(f"MM SELL L{i+1}: {level_sell_size} {product} at {sell_price}")
        
        return orders

    def trade_kelp(self, product: str, state: TradingState) -> List[Order]:
        """Strategy for Kelp trading - placeholder for now"""
        # This is just a placeholder - implement your KELP strategy here
        return []