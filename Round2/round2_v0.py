from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any, Tuple
import statistics
import jsonpickle
import math
import pandas as pd
import numpy as np
import json
from collections import defaultdict


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

class Product:
    KELP = "KELP"
    RAINFORST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBE = "DJEMBE"
    PICNIC_BASKET1 = "PICNIC_BASKET1",
    PICNIC_BASKET2 = "PICNIC_BASKET2"

parameters = {
    Product.RAINFORST_RESIN:{
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,# we won't penny at 9999 or 10001
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10, #after we hit this, try and reduce back down to 0 by taking less edge on each trade

    },
    Product.KELP: {
        "take_width": 1, 
        "clear_width": 0,
        "disregard_edge" : 1, 
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.6438,
        "join_edge": 0, #never join orders, always try and penny
        "default_edge" : 1, 

    },
    Product.SQUID_INK: {
        "fair_value": 2000,  #? not sure
        "take_width": 1,  # we shouldn't be making two side markets
        "clear_width": 0,
        "disregard_edge" : 1, 
        "prevent_adverse": True,
        "adverse_volume": 15,
        "join_edge": 0, #never join orders, always try and penny
        "default_edge" : 1, 

    },
    Product.PICNIC_BASKET1: {
        "arb_width": 0,
        "fair_value": None, #UPDATE LATER
        #ADD IN MM PARAMS
    },
    Product.PICNIC_BASKET2:{
        "arb_width": 0,
    },
    Product.CROISSANTS: {
        "take_width": 1,
    },
    Product.JAMS: {
        "take_width": 1,
    },
    Product.DJEMBE: {
        "take_width":1,
    }
}
class Trader:
    def __init__(self, params = None):
        # Initialize trader state
        if params is None:
            params = parameters
        self.params = params 
        self.LIMIT = {Product.RAINFORST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50, Product.CROISSANTS: 250, Product.JAMS: 350, Product.DJEMBE: 60, Product.PICNIC_BASKET1: 60, Product.PICNIC_BASKET2: 100}

    def take_best_orders(self,
                         product: str,
                         fair_value: int,
                         take_width: float, 
                         orders: List[Order],
                         order_depth: OrderDepth,
                         position: int,
                         buy_order_volume: int,
                         sell_order_volume: int,
                         prevent_adverse: bool=False,
                         adverse_volume: int = 0) -> (int,int):
        position_limit = self.LIMIT[product]
        # find any orders below fair value and lift them, update our buy quantity 
        if(len(order_depth.sell_orders) != 0):
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -1 *order_depth.sell_orders[best_ask]

        if not prevent_adverse or abs(best_ask_volume) < adverse_volume: #if we're not only looking at large MMs or if this trade was not from a MM (we shouldn't lift the MMs)
            if best_ask < fair_value - take_width:
                quantity = min(best_ask_volume, position_limit - position)
                if quantity >0 :
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity # so we nullify that trade in the book since we've already lifted it
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask] #if we filled the trade (which means it was within our position limit), delete it
        #if anyone is willing to buy for more than the fair value, hit them and update our sell quantity
        if(len(order_depth.buy_orders) != 0 ):
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
        if not prevent_adverse or abs(best_bid_volume) < adverse_volume:
            if best_bid > fair_value + take_width:
                quantity = min(best_bid_volume, position_limit + position) #since we are selling, our amount that can be sold is pos - (-pos limit)
                if quantity > 0:
                    orders.append(Order(product,best_bid,-1*quantity))
                    sell_order_volume += quantity 
                    order_depth.buy_orders[best_bid] -= quantity 
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume
    
    def market_make(self,
                    product: str,
                    orders: List[Order],
                    bid: int,
                    ask: int,
                    position: int,
                    buy_order_volume: int,
                    sell_order_volume: int,
                    ) -> (int,int):
        # fill remaining trades with market making strategy
        buy_quantity = self.LIMIT[product] - position - buy_order_volume 
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume
    
    def clear_position_order(self,
            product :str,
            fair_value: float,
            width: int,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,)-> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask =round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] - (position)
        if (position_after_take > 0):
            #find all existing buyers in the markets with bids > fair + sigma
            clear_quantity = sum(volume for price,volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity,clear_quantity)
            if sent_quantity > 0: #place these
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if (position_after_take < 0):
            clear_quantity = sum(volume for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity >0: #place these
                orders.append(Order(product,fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume
    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if(len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) !=  0):
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -1 *order_depth.sell_orders[best_ask]
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"] ]
            mm_ask = min(filtered_ask) if len(filtered_ask) >0 else None 
            mm_bid = min(filtered_bid) if len(filtered_bid) > 0 else None 
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2 # if no last price (first instance)
                else: 
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) /2 

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price 
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta" ] # mean reversion 

                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price 
            traderObject["kelp_last_price"] = mmmid_price
            return fair 
        return None 
    def take_orders( # we can repeatedly take orders with this method until our position limits are hit or there are no more favourable trades in the market
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            take_width: float,
            position: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,) -> (List[Order], int,int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0 
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume,sell_order_volume
    def clear_orders( # clear the orders after each take with an empty order list
            self,
            product:str,
            order_depth: OrderDepth,
            fair_value: float,
            clear_width: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    )-> (List[Order], int,int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders,buy_order_volume, sell_order_volume
    def make_orders( #execute market making strategy
            self,
            product:str,
            order_depth: OrderDepth,
            fair_value: float,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            disregard_edge: float, #don't penny or join within this edge
            join_edge: float, # join in / match these trades
            default_edge: float, # default if there are no levels to join in or improve at

    ):
        orders: List[Order] = []
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge
        ]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None 
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None 

        ask = round( fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair # we match them if they are within the join edge

            else: # use penny 
                ask = best_ask_above_fair - 1 
        
        bid = round( fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair # we match them if they are within the join edge

            else: # use penny 
                bid = best_bid_below_fair +1 
            
        buy_order_volume, sell_order_volume = self.market_make( #actually do thsoe trades
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def find_arb(self,
                 constituents: Dict[str, int], #each constituent product and the amount, recipe for making the basket
                 basket: str,
                 order_depth_basket: OrderDepth,
                 order_depth_constituents: Dict[str, OrderDepth], #keys are the constituent products, values are the  
                 position_basket: int,
                 positions_constituent: dict[str, int], #use a dictionary?
                 min_arb_width: int = 0,
                 
                 ) -> List[(List[Order], int)]: #returns a list of all possible arbs with the PnL from those Arbs.
        buy_quantity_basket = self.LIMIT[basket] - position_basket
        sell_quantity_basket = self.LIMIT[basket] + position_basket
        constituents_buy_sell_dict = {product: (self.LIMIT[product] - positions_constituent[product],
                                                                             self.LIMIT[product] + positions_constituent[product]) for product in positions_constituent.keys() }
        #buy basket and sell constituents if basket cost < sum constituents
        #RIGHT NOW I AM ONLY LOOKING AT THE BEST BID, THAT IS THERE NEEDS TO BE AT LEAST ENOUGH OF THE BEST BID OF EACH CONSTITUENT TO FORM A BASKET
        #LATER CAN BE UPDATED TO LOOK UP THE ORDER BOOK
        orders: List[Order] = []
        if len(order_depth_basket.sell_orders) > 0 and (all(map(lambda x : (len(x.sell_orders) > 0), order_depth_constituents.values()))):
            best_ask_basket = min(order_depth_basket.sell_orders.keys())
            best_ask_basket_vol = -1* order_depth_basket[best_ask_basket]
            best_bids = {item: (max(order_depth_item.buy_orders)) for item, order_depth_item in order_depth_constituents.items()}
            best_bid_volumes = {constituent: order_depth_item[best_bids[constituent]] for constituent, order_depth_item in order_depth_constituents.items()}
            baskets_that_can_be_made = 1 #IMPLEMENT THIS
            arbs = []
            if all([best_bid_volumes[constituent] >= constituents[constituent] for constituent in best_bid_volumes.keys()]):# have enough bids to form a basket for each constituent
                #CHECK IF ARB
                arb_pnl = sum([best_bids[constituent]*constituents[constituent] for constituent in constituents.keys()]) - best_ask_basket
                if(arb_pnl > 0):
                    #right now just doing one arb, but implement as many if possible later with baskets_that_can_be_made.
                    orders.append(Order(basket, best_ask_basket, baskets_that_can_be_made)) # UPDATE QUANTITY
                    orders.extend([Order(constituent, best_bids[constituent], -1*constituents[constituent]*baskets_that_can_be_made) for constituent in constituents.keys()])
                    buy_quantity_basket += baskets_that_can_be_made
                    for constituent in constituents.keys():
                        constituents_buy_sell_dict += -1*constituents[constituent]*baskets_that_can_be_made 
                    arbs.append(,arb_pnl) #SO CONFUSED WHAT TO RETURN

        return 
    def clear_arb(self,
                  buySellOrderVolume: dict[(int,int)],
                  order_depth: OrderDepth,
                  position_basket: int,
                  positions_constituent: dict[int],
                  clear_width: int,
                  fair_values: dict[int],) -> (List[Order], dict[int]):
        return NotImplementedError
    def run(self, state: TradingState):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Initialize the result dict with empty lists for all products
        result = {}
        if Product.RAINFORST_RESIN in self.params and Product.RAINFORST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFORST_RESIN]
                if Product.RAINFORST_RESIN in state.position 
                else 0 
            )
        resin_take_orders, buy_order_volume, sell_order_volume = (
            self.take_orders(
                Product.RAINFORST_RESIN, 
                state.order_depths[Product.RAINFORST_RESIN],
                self.params[Product.RAINFORST_RESIN]["fair_value"],
                self.params[Product.RAINFORST_RESIN]["take_width"],
                resin_position
            )
        )
        resin_clear_orders, buy_order_volume, sell_order_volume = (
            self.clear_orders(
                Product.RAINFORST_RESIN,
                state.order_depths[Product.RAINFORST_RESIN],
                self.params[Product.RAINFORST_RESIN]["fair_value"],
                self.params[Product.RAINFORST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
        )
        resin_make_orders, _,_ = self.make_orders(
            Product.RAINFORST_RESIN,
                state.order_depths[Product.RAINFORST_RESIN],
                self.params[Product.RAINFORST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFORST_RESIN]["disregard_edge"],
                self.params[Product.RAINFORST_RESIN]["join_edge"],
                self.params[Product.RAINFORST_RESIN]["default_edge"],
        )
        result[Product.RAINFORST_RESIN] = (
            resin_take_orders + resin_clear_orders + resin_make_orders # take then clear then make
        )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = ( kelp_take_orders + kelp_clear_orders + kelp_make_orders)
        result[Product.SQUID_INK] = []
        
        # We're not using conversions in this implementation
        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    