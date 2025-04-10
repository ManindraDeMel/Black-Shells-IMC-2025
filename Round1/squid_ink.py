from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    def __init__(self):
        self.squid_ink_prices = []  # Track mid prices for squid ink
    
    def run(self, state: TradingState):
        """
        Market making strategy focusing exclusively on squid ink
        """
        # Initialize the result dict
        result = {}
        
        # Check if squid ink is in the available products
        if "SQUID_INK" in state.order_depths:
            order_depth = state.order_depths["SQUID_INK"]
            
            # Initialize orders list for squid ink
            orders: List[Order] = []
            
            # Get current position in squid ink (defaulting to 0 if not present)
            squid_ink_position = state.position.get("SQUID_INK", 0)
            
            # LOOK THROUGH BIDS AND ASKS, FIND THE BEST BID (highest) AND THE best ASK (lowest) WITH VOLUME > 20
            best_bid_price = None
            best_ask_price = None
            
            # Process buy orders (bids)
            if len(order_depth.buy_orders) > 0:
                # Sort bids in descending order (highest first)
                sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                
                # Find the best bid with volume > 20
                for price, volume in sorted_bids:
                    if volume > 20:
                        best_bid_price = price
                        break
            
            # Process sell orders (asks)
            if len(order_depth.sell_orders) > 0:
                # Sort asks in ascending order (lowest first)
                sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                
                # Find the best ask with volume > 20
                for price, volume in sorted_asks:
                    if abs(volume) > 20:  # Volume is negative for sell orders
                        best_ask_price = price
                        break
            
            # CALCULATE THE FAIR VALUE AS THE AVERAGE of that bid price and ask price
            mm_mid_price = None
            if best_bid_price is not None and best_ask_price is not None:
                mm_mid_price = (best_bid_price + best_ask_price) / 2
                # Track the mid price
                self.squid_ink_prices.append(mm_mid_price)
            elif len(self.squid_ink_prices) > 0:
                # Use the last known mid price if current one can't be calculated
                mm_mid_price = self.squid_ink_prices[-1]
            else:
                # Fallback to simple mid price calculation if no history
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mm_mid_price = (best_bid + best_ask) / 2
                    self.squid_ink_prices.append(mm_mid_price)
            
            # TIGHTEN THAT MARKET BY 1 OR 2 AND MARKET MAKE
            if mm_mid_price is not None:
                # Define spread tightening
                spread_tightening = 1
                
                # Calculate our buy and sell prices around the fair value
                our_bid = int(mm_mid_price) - spread_tightening
                our_ask = int(mm_mid_price) + spread_tightening
                
                # Position limit for squid ink (typically 50 for most products)
                position_limit = 50
                
                # Determine volume to trade based on position limits
                buy_volume = min(10, position_limit - squid_ink_position)  # Limit to 10 units per order
                sell_volume = min(10, position_limit + squid_ink_position)  # Limit to 10 units per order
                
                # Only place buy orders if we have room to buy
                if buy_volume > 0:
                    orders.append(Order("SQUID_INK", our_bid, buy_volume))
                
                # Only place sell orders if we have room to sell
                if sell_volume > 0:
                    orders.append(Order("SQUID_INK", our_ask, -sell_volume))
            
            # Add orders to the result
            result["SQUID_INK"] = orders
        
        # No conversions needed for squid ink
        conversions = 0
        
        # String to hold trader state data (we'll serialize the mid price list)
        traderData = str(self.squid_ink_prices)
        
        return result, conversions, traderData