from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    def __init__(self):
        self.kelp_prices = []  # Track mid prices for kelp
    
    def run(self, state: TradingState):
        """
        Market making strategy focusing exclusively on kelp with position clearing
        """
        # Initialize the result dict
        result = {}
        
        # Check if kelp is in the available products
        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            
            # Initialize orders list for kelp
            orders: List[Order] = []
            
            # Get current position in kelp (defaulting to 0 if not present)
            kelp_position = state.position.get("KELP", 0)
            
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
                self.kelp_prices.append(mm_mid_price)
            elif len(self.kelp_prices) > 0:
                # Use the last known mid price if current one can't be calculated
                mm_mid_price = self.kelp_prices[-1]
            else:
                # Fallback to simple mid price calculation if no history
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mm_mid_price = (best_bid + best_ask) / 2
                    self.kelp_prices.append(mm_mid_price)
            
            # Position clearing logic
            position_limit = 50
            aggressive_clear_threshold = position_limit * 0.7  # When to start aggressively reducing position
            
            # TIGHTEN THAT MARKET BY 1 OR 2 AND MARKET MAKE
            if mm_mid_price is not None:
                # Define spread tightening
                spread_tightening = 1
                
                # Calculate our buy and sell prices around the fair value
                our_bid = int(mm_mid_price) - spread_tightening
                our_ask = int(mm_mid_price) + spread_tightening
                
                # Position clearing logic - adjust prices if we need to clear positions
                if kelp_position > aggressive_clear_threshold:
                    # We have a large long position, make selling more attractive
                    our_ask = our_ask - 1  # More aggressive sell price
                    
                    # Look for opportunities to clear positions at favorable prices
                    if len(order_depth.buy_orders) > 0:
                        # Sort bids in descending order (highest first)
                        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                        for price, volume in sorted_bids:
                            # If we can sell at or above our calculated fair value, do it immediately
                            if price >= mm_mid_price and kelp_position > 0:
                                clear_volume = min(volume, kelp_position)
                                orders.append(Order("KELP", price, -clear_volume))
                                kelp_position -= clear_volume
                
                elif kelp_position < -aggressive_clear_threshold:
                    # We have a large short position, make buying more attractive
                    our_bid = our_bid + 1  # More aggressive buy price
                    
                    # Look for opportunities to clear positions at favorable prices
                    if len(order_depth.sell_orders) > 0:
                        # Sort asks in ascending order (lowest first)
                        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                        for price, volume in sorted_asks:
                            # If we can buy at or below our calculated fair value, do it immediately
                            if price <= mm_mid_price and kelp_position < 0:
                                clear_volume = min(abs(volume), abs(kelp_position))
                                orders.append(Order("KELP", price, clear_volume))
                                kelp_position += clear_volume
                
                # Determine volume to trade based on position limits and current position
                buy_volume = min(10, position_limit - kelp_position)  # Limit to 10 units per order
                sell_volume = min(10, position_limit + kelp_position)  # Limit to 10 units per order
                
                # Adjust volumes based on current position
                if kelp_position > 0:
                    # We're already long, reduce buy volume and increase sell volume
                    buy_volume = max(0, buy_volume - int(kelp_position/5))
                    sell_volume = min(position_limit, sell_volume + int(kelp_position/3))
                elif kelp_position < 0:
                    # We're already short, increase buy volume and reduce sell volume
                    buy_volume = min(position_limit, buy_volume + int(abs(kelp_position)/3))
                    sell_volume = max(0, sell_volume - int(abs(kelp_position)/5))
                
                # Only place buy orders if we have room to buy and haven't filled our clearing orders above
                if buy_volume > 0:
                    orders.append(Order("KELP", our_bid, buy_volume))
                
                # Only place sell orders if we have room to sell and haven't filled our clearing orders above
                if sell_volume > 0:
                    orders.append(Order("KELP", our_ask, -sell_volume))
            
            # Add orders to the result
            result["KELP"] = orders
        
        # No conversions needed for kelp
        conversions = 0
        
        # String to hold trader state data (we'll serialize the mid price list)
        traderData = str(self.kelp_prices)
        
        return result, conversions, traderData