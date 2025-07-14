import jsonpickle
import math
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

from enum import IntEnum
from abc import ABC, abstractmethod

HARD_POSITION_LIMIT = 50

PARAMS = {
    "RAINFOREST_RESIN": {
        "fair_value": 10_000,
        "position_limit": 30,
        "take_width": 0.5,
        "default_edge": 1.0,
        "join_edge": 1,
        "disregard_edge": 0.5,
    },
    "SQUID_INK": {
        "spread_alpha": 1037.39,
        "spread_beta": 0.0885,
        "position_limit": 50,
        "z_entry": 2,
        "z_exit": 0.5,
        "z_window": 1000
    },
    "KELP": {
        "take_width": 1.5,
        "clear_width": 1,
        "reversion_beta": -0.1,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 0.5,
        "soft_position_limit": 50,
        "adverse_volume": 12,
        "base_fair": 2000,  # only fallback
        "position_limit": 50
    },
    "PICNIC_BASKET1": {
        "weights": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
        "zscore_threshold": 2.0,
        "default_spread_mean": 0,
        "spread_std_window": 50,
        "position_limit": 60
    },
    "PICNIC_BASKET2": {
        "weights": {"CROISSANTS": 4, "JAMS": 2},
        "zscore_threshold": 2.0,
        "default_spread_mean": 0,
        "spread_std_window": 50,
        "position_limit": 100
    },
    "DJEMBES": {
        "position_limit": 60,
        "make_edge": 1.0,
        "take_edge": 0.5
    },
    "VOLCANIC_ROCK": {
    "position_limit": 400
    },
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "strike": 9500,
        "position_limit": 200,
        "expiry_days": 7,
        "iv_curve_fit_window": 50,
        "min_iv_threshold": 0.0001,
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "strike": 9750,
        "position_limit": 200,
        "expiry_days": 7,
        "iv_curve_fit_window": 50,
        "min_iv_threshold": 0.0001,
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "strike": 10000,
        "position_limit": 200,
        "expiry_days": 7,
        "iv_curve_fit_window": 50,
        "min_iv_threshold": 0.0001,
    },
    "VOLCANIC_ROCK_VOUCHER_10250": {
        "strike": 10250,
        "position_limit": 200,
        "expiry_days": 7,
        "iv_curve_fit_window": 50,
        "min_iv_threshold": 0.0001,
    },
    "VOLCANIC_ROCK_VOUCHER_10500": {
        "strike": 10500,
        "position_limit": 200,
        "expiry_days": 7,
        "iv_curve_fit_window": 50,
        "min_iv_threshold": 0.0001,
    },
    "MAGNIFICENT_MACARONS": {
        "position_limit": 75,
        "conversion_limit": 10,
        "take_width": 1.5,
        "clear_width": 1,
        "reversion_beta": -0.1,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 0.5,
        "soft_position_limit": 50,
        "adverse_volume": 12,
        "CriticalSunlightIndex": 55, #50
    }
}

# --- LOGGER ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        
        # Clean spread histories in trader_data before logging
        def recursively_round_floats(obj, sigfigs=4):
            if isinstance(obj, float):
                return float(f"{obj:.{sigfigs}g}")
            elif isinstance(obj, list):
                return [recursively_round_floats(x, sigfigs) for x in obj]
            elif isinstance(obj, dict):
                return {k: recursively_round_floats(v, sigfigs) for k, v in obj.items()}
            return obj
        
        cleaned_trader_data = recursively_round_floats(jsonpickle.decode(trader_data))
        
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(jsonpickle.encode(cleaned_trader_data), max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [d.buy_orders, d.sell_orders] for s, d in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for ts in state.own_trades.values() for t in ts],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for ts in state.market_trades.values() for t in ts],
            state.position,
            [
                state.observations.plainValueObservations,
                {
                    p: [
                        o.bidPrice, o.askPrice, o.transportFees,
                        o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex
                    ] for p, o in state.observations.conversionObservations.items()
                }
            ]
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for ol in orders.values() for o in ol]

    def to_json(self, value: Any) -> str:
        return jsonpickle.encode(value, unpicklable=False)

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

# Logger Utility

def round_list(values: List[float], sigfigs: int = 4) -> List[float]:
    def round_sig(x):
        return float(f"{x:.{sigfigs}g}") if x is not None else None
    return [round_sig(v) for v in values]

# --- RESIN STRATEGY ---
class ResinStrategy:
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self.cash = 0

    def run(self, state: TradingState) -> List[Order]:
        p = self.params
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        orders: List[Order] = []

        best_bids = sorted(order_depth.buy_orders.items(), reverse=True)
        best_asks = sorted(order_depth.sell_orders.items())

        max_buy = HARD_POSITION_LIMIT - position
        max_sell = HARD_POSITION_LIMIT + position

        for ask, vol in best_asks:
            if ask < p["fair_value"] - p["take_width"] and max_buy > 0:
                qty = min(-vol, max_buy)
                orders.append(Order(self.symbol, ask, qty))
                max_buy -= qty

        for bid, vol in best_bids:
            if bid > p["fair_value"] + p["take_width"] and max_sell > 0:
                qty = min(vol, max_sell)
                orders.append(Order(self.symbol, bid, -qty))
                max_sell -= qty

        ask_quote = round(p["fair_value"] + p["default_edge"])
        bid_quote = round(p["fair_value"] - p["default_edge"])

        if best_asks:
            best_ask = best_asks[0][0]
            if best_ask > p["fair_value"] + p["disregard_edge"]:
                ask_quote = best_ask - 1 if best_ask - p["fair_value"] > p["join_edge"] else best_ask

        if best_bids:
            best_bid = best_bids[0][0]
            if best_bid < p["fair_value"] - p["disregard_edge"]:
                bid_quote = best_bid + 1 if p["fair_value"] - best_bid > p["join_edge"] else best_bid

        if position > p["position_limit"]:
            ask_quote -= 1
        elif position < -p["position_limit"]:
            bid_quote += 1

        mm_buy_qty = min(p["position_limit"] - position, HARD_POSITION_LIMIT - position)
        mm_sell_qty = min(p["position_limit"] + position, HARD_POSITION_LIMIT + position)

        if mm_buy_qty > 0:
            orders.append(Order(self.symbol, bid_quote, mm_buy_qty))
        if mm_sell_qty > 0:
            orders.append(Order(self.symbol, ask_quote, -mm_sell_qty))

        for trade in state.own_trades.get(self.symbol, []):
            self.cash += trade.quantity * (trade.price if trade.seller == "SUBMISSION" else -trade.price)

        mid = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
        pnl = self.cash + position * mid

        logger.print(f"[{state.timestamp}] {self.symbol} | Pos: {position} | Mid: {mid:.2f} | PnL: {pnl:.2f}")
        return orders

    def save(self):
        return {"cash": self.cash}

    def load(self, data):
        self.cash = data.get("cash", 0)

# --- SQUID INK STRATEGY ---
class SquidInkStrategy:
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self.position = 0
        self.spread_history = []
        self.alpha = self.params["spread_alpha"]
        self.beta = self.params["spread_beta"]
        self.z_window = self.params["z_window"]
        self.z_entry = self.params["z_entry"]
        self.z_exit = self.params["z_exit"]

    def run(self, state: TradingState) -> List[Order]:
        if "RAINFOREST_RESIN" not in state.order_depths:
            return []

        order_depth = state.order_depths[self.symbol]
        resin_depth = state.order_depths["RAINFOREST_RESIN"]
        orders = []

        if not order_depth.buy_orders or not order_depth.sell_orders or not resin_depth.buy_orders or not resin_depth.sell_orders:
            return []

        squid_bid = max(order_depth.buy_orders)
        squid_ask = min(order_depth.sell_orders)
        squid_mid = (squid_bid + squid_ask) / 2

        resin_bid = max(resin_depth.buy_orders)
        resin_ask = min(resin_depth.sell_orders)
        resin_mid = (resin_bid + resin_ask) / 2

        fair_price = self.alpha + self.beta * resin_mid
        spread = squid_mid - fair_price

        self.spread_history.append(spread)
        if len(self.spread_history) > self.z_window:
            self.spread_history.pop(0)

        z = 0
        if len(self.spread_history) >= 10:
            std = np.std(self.spread_history)
            mean = np.mean(self.spread_history)
            z = (spread - mean) / std if std > 0 else 0

        position = state.position.get(self.symbol, 0)
        max_buy = HARD_POSITION_LIMIT - position
        max_sell = HARD_POSITION_LIMIT + position

        if z > self.z_entry and position > -HARD_POSITION_LIMIT:
            qty = min(order_depth.buy_orders[squid_bid], max_sell)
            orders.append(Order(self.symbol, squid_bid, -qty))

        elif z < -self.z_entry and position < HARD_POSITION_LIMIT:
            qty = min(-order_depth.sell_orders[squid_ask], max_buy)
            orders.append(Order(self.symbol, squid_ask, qty))

        elif abs(z) < self.z_exit:
            if position > 0:
                orders.append(Order(self.symbol, squid_bid, -position))
            elif position < 0:
                orders.append(Order(self.symbol, squid_ask, -position))

        logger.print(f"[{state.timestamp}] SQUID_INK | Fair: {fair_price:.2f} | Spread: {spread:.2f} | Z: {z:.2f} | Pos: {position}")
        return orders

    def save(self):
        return {
            "position": self.position,
            "spread_history": self.spread_history
        }

    def load(self, data):
        self.position = data.get("position", 0)
        self.spread_history = data.get("spread_history", [])

# --- KELP STRATEGY ---
class KelpStrategy:
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self.last_fair = None
        self.cash = 0

    def get_fair_price(self, order_depth: OrderDepth):
        p = self.params
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders

        filtered_bids = [price for price, vol in bids.items() if vol >= p["adverse_volume"]]
        filtered_asks = [price for price, vol in asks.items() if -vol >= p["adverse_volume"]]

        if filtered_bids and filtered_asks:
            mm_mid = (max(filtered_bids) + min(filtered_asks)) / 2
        elif bids and asks:
            mm_mid = (max(bids.keys()) + min(asks.keys())) / 2
        else:
            mm_mid = self.last_fair if self.last_fair else 2000  # fallback to default

        if self.last_fair is not None:
            fair = mm_mid + p["reversion_beta"] * (mm_mid - self.last_fair)
        else:
            fair = mm_mid

        self.last_fair = fair
        return fair

    def run(self, state: TradingState) -> List[Order]:
        p = self.params
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        orders = []

        fair = self.get_fair_price(order_depth)

        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())

        max_buy = HARD_POSITION_LIMIT - position
        max_sell = HARD_POSITION_LIMIT + position

        # --- Take ---
        for ask, vol in asks:
            if ask < fair - p["take_width"] and max_buy > 0:
                qty = min(-vol, max_buy)
                orders.append(Order(self.symbol, ask, qty))
                max_buy -= qty

        for bid, vol in bids:
            if bid > fair + p["take_width"] and max_sell > 0:
                qty = min(vol, max_sell)
                orders.append(Order(self.symbol, bid, -qty))
                max_sell -= qty

        # --- Clear ---
        position_after_take = position + (HARD_POSITION_LIMIT - max_buy) - (HARD_POSITION_LIMIT - max_sell)
        ask_clear = round(fair + p["clear_width"])
        bid_clear = round(fair - p["clear_width"])

        if position_after_take > 0:
            qty = min(position_after_take, max_sell)
            if qty > 0:
                orders.append(Order(self.symbol, ask_clear, -qty))
                max_sell -= qty

        if position_after_take < 0:
            qty = min(-position_after_take, max_buy)
            if qty > 0:
                orders.append(Order(self.symbol, bid_clear, qty))
                max_buy -= qty

        # --- Market Make ---
        bid_quote = round(fair - p["default_edge"])
        ask_quote = round(fair + p["default_edge"])

        if bids:
            best_bid = bids[0][0]
            if best_bid < fair - p["disregard_edge"]:
                bid_quote = best_bid + 1 if fair - best_bid > p["join_edge"] else best_bid

        if asks:
            best_ask = asks[0][0]
            if best_ask > fair + p["disregard_edge"]:
                ask_quote = best_ask - 1 if best_ask - fair > p["join_edge"] else best_ask

        # Position-based adjustment
        if position > p["soft_position_limit"]:
            ask_quote -= 1
        elif position < -p["soft_position_limit"]:
            bid_quote += 1

        mm_buy_qty = min(p["soft_position_limit"] - position, max_buy)
        mm_sell_qty = min(p["soft_position_limit"] + position, max_sell)

        if mm_buy_qty > 0:
            orders.append(Order(self.symbol, bid_quote, mm_buy_qty))
        if mm_sell_qty > 0:
            orders.append(Order(self.symbol, ask_quote, -mm_sell_qty))

        # Track PnL
        for trade in state.own_trades.get(self.symbol, []):
            self.cash += trade.quantity * (trade.price if trade.seller == "SUBMISSION" else -trade.price)

        mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else fair
        pnl = self.cash + position * mid
        logger.print(f"[{state.timestamp}] {self.symbol} | Fair: {fair:.2f} | Pos: {position} | PnL: {pnl:.2f}")

        return orders

    def save(self):
        return {"cash": self.cash, "last_fair": self.last_fair}

    def load(self, data):
        self.cash = data.get("cash", 0)
        self.last_fair = data.get("last_fair", None)
        
class SpreadTracker:
    def __init__(self, mean: float = 0, std_window: int = 50):
        self.spread_history: List[float] = []
        self.default_mean = mean
        self.std_window = std_window

    def update_and_compute_z(self, spread: float) -> float:
        self.spread_history.append(spread)
        if len(self.spread_history) > self.std_window:
            self.spread_history.pop(0)
        std = np.std(self.spread_history)
        return (spread - self.default_mean) / std if std > 0 else 0

class Basket1Strategy:
    def __init__(self, basket: str, params: dict):
        self.basket = basket
        self.weights = params["weights"]
        self.z_threshold = params["zscore_threshold"]
        self.limit = params["position_limit"]
        self.tracker = SpreadTracker(params["default_spread_mean"], params["spread_std_window"])

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2
        return None

    def compute_spread(self, state: TradingState) -> float:
        basket_depth = state.order_depths.get(self.basket)
        if not basket_depth:
            return 0
        basket_mid = self.get_mid_price(basket_depth)
        if basket_mid is None:
            return 0
        synthetic = 0
        for comp, weight in self.weights.items():
            comp_depth = state.order_depths.get(comp)
            if not comp_depth:
                return 0
            comp_mid = self.get_mid_price(comp_depth)
            if comp_mid is None:
                return 0
            synthetic += weight * comp_mid
        return basket_mid - synthetic

    def run(self, state: TradingState) -> List[Order]:
        position = state.position.get(self.basket, 0)
        order_depth = state.order_depths.get(self.basket)
        if not order_depth:
            return []

        spread = self.compute_spread(state)
        z = self.tracker.update_and_compute_z(spread)
        orders = []

        if z < -self.z_threshold and position < self.limit:
            best_ask = min(order_depth.sell_orders)
            qty = min(-order_depth.sell_orders[best_ask], self.limit - position)
            if qty > 0:
                orders.append(Order(self.basket, best_ask, qty))

        elif z > self.z_threshold and position > -self.limit:
            best_bid = max(order_depth.buy_orders)
            qty = min(order_depth.buy_orders[best_bid], self.limit + position)
            if qty > 0:
                orders.append(Order(self.basket, best_bid, -qty))

        logger.print(f"[{state.timestamp}] {self.basket} | Spread: {spread:.2f} | Z: {z:.2f} | Pos: {position}")
        return orders

    def save(self):
        return {"spread_history": self.tracker.spread_history}

    def load(self, data):
        if data and "spread_history" in data:
            self.tracker.spread_history = data["spread_history"]
            
class Basket2Strategy:
    def __init__(self, basket: str, params: dict):
        self.basket = basket
        self.weights = params["weights"]
        self.z_threshold = params["zscore_threshold"]
        self.limit = params["position_limit"]
        self.tracker = SpreadTracker(params["default_spread_mean"], params["spread_std_window"])

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2
        return None

    def compute_spread(self, state: TradingState) -> float:
        basket_depth = state.order_depths.get(self.basket)
        if not basket_depth:
            return 0
        basket_mid = self.get_mid_price(basket_depth)
        if basket_mid is None:
            return 0
        synthetic = 0
        for comp, weight in self.weights.items():
            comp_depth = state.order_depths.get(comp)
            if not comp_depth:
                return 0
            comp_mid = self.get_mid_price(comp_depth)
            if comp_mid is None:
                return 0
            synthetic += weight * comp_mid
        return basket_mid - synthetic

    def run(self, state: TradingState) -> List[Order]:
        position = state.position.get(self.basket, 0)
        order_depth = state.order_depths.get(self.basket)
        if not order_depth:
            return []

        spread = self.compute_spread(state)
        z = self.tracker.update_and_compute_z(spread)
        orders = []

        if z < -self.z_threshold and position < self.limit:
            best_ask = min(order_depth.sell_orders)
            qty = min(-order_depth.sell_orders[best_ask], self.limit - position)
            if qty > 0:
                orders.append(Order(self.basket, best_ask, qty))

        elif z > self.z_threshold and position > -self.limit:
            best_bid = max(order_depth.buy_orders)
            qty = min(order_depth.buy_orders[best_bid], self.limit + position)
            if qty > 0:
                orders.append(Order(self.basket, best_bid, -qty))

        logger.print(f"[{state.timestamp}] {self.basket} | Spread: {spread:.2f} | Z: {z:.2f} | Pos: {position}")
        return orders

    def save(self):
        return {"spread_history": self.tracker.spread_history}

    def load(self, data):
        if data and "spread_history" in data:
            self.tracker.spread_history = data["spread_history"]

class CrossBasketArbitrage:
    def __init__(self):
        self.basket1 = "PICNIC_BASKET1"
        self.basket2 = "PICNIC_BASKET2"
        self.djembe = "DJEMBES"
        self.position_limit = 50
        self.zscore_threshold = 2.0
        self.default_mean = 0
        self.window = 50
        self.tracker = SpreadTracker(self.default_mean, self.window)

    def get_mid(self, order_depth: OrderDepth):
        if order_depth and order_depth.buy_orders and order_depth.sell_orders:
            return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2
        return None

    def compute_spread(self, order_depths: Dict[str, OrderDepth]):
        b1 = self.get_mid(order_depths.get(self.basket1))
        b2 = self.get_mid(order_depths.get(self.basket2))
        djembe = self.get_mid(order_depths.get(self.djembe))
        if None in (b1, b2, djembe):
            return None
        return 2 * b1 - (3 * b2 + 2 * djembe)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {self.basket1: [], self.basket2: [], self.djembe: []}
        order_depths = state.order_depths
        positions = state.position

        spread = self.compute_spread(order_depths)
        if spread is None:
            return orders

        z = self.tracker.update_and_compute_z(spread)
        pos1 = positions.get(self.basket1, 0)
        pos2 = positions.get(self.basket2, 0)
        pos_dj = positions.get(self.djembe, 0)

        vol1 = self.position_limit - abs(pos1)
        vol2 = self.position_limit - abs(pos2)
        vol_dj = self.position_limit - abs(pos_dj)
        max_units = min(vol1 // 2, vol2 // 3, vol_dj // 2)

        if max_units <= 0:
            return orders

        if z < -self.zscore_threshold:
            # Long LHS, short RHS
            b1_ask = min(order_depths[self.basket1].sell_orders)
            b2_bid = max(order_depths[self.basket2].buy_orders)
            dj_bid = max(order_depths[self.djembe].buy_orders)

            orders[self.basket1].append(Order(self.basket1, b1_ask, 2 * max_units))
            orders[self.basket2].append(Order(self.basket2, b2_bid, -3 * max_units))
            orders[self.djembe].append(Order(self.djembe, dj_bid, -2 * max_units))

        elif z > self.zscore_threshold:
            # Short LHS, long RHS
            b1_bid = max(order_depths[self.basket1].buy_orders)
            b2_ask = min(order_depths[self.basket2].sell_orders)
            dj_ask = min(order_depths[self.djembe].sell_orders)

            orders[self.basket1].append(Order(self.basket1, b1_bid, -2 * max_units))
            orders[self.basket2].append(Order(self.basket2, b2_ask, 3 * max_units))
            orders[self.djembe].append(Order(self.djembe, dj_ask, 2 * max_units))

        logger.print(f"[{state.timestamp}] Cross Basket Z: {z:.2f} | Spread: {spread:.2f}")
        return orders

# --- VOLCANIC_ROCK_VOUCHER Strategy ---
class SmileFitter:
    def __init__(self, chunk_size, default_coeffs):
        self.chunk_size = chunk_size
        self.default_coeffs = default_coeffs
        self.data_by_chunk = {}  # mapping chunk index -> (m, iv) pairs
        self.coefficients_by_chunk = {}  # chunk -> (a, b, c)
        self.current_chunk = 0

    def _get_chunk(self, timestamp: int) -> int:
        return timestamp // self.chunk_size

    def add_observation(self, m: float, iv: float, timestamp: int):
        if not (np.isfinite(m) and np.isfinite(iv)):
            return

        chunk = self._get_chunk(timestamp)
        if chunk not in self.data_by_chunk:
            self.data_by_chunk[chunk] = []

        self.data_by_chunk[chunk].append((m, iv))

        # Refit only for previous chunk (to be used by current)
        prev_chunk = chunk - 1
        if prev_chunk not in self.coefficients_by_chunk and prev_chunk in self.data_by_chunk:
            self.refit_smile(prev_chunk)

        self.current_chunk = chunk

    def refit_smile(self, chunk: int):
        data = self.data_by_chunk.get(chunk, [])
        if len(data) < 10:
            self.coefficients_by_chunk[chunk] = self.default_coeffs
            return

        m_vals, iv_vals = zip(*data)
        mean = np.mean(iv_vals)
        std = np.std(iv_vals)
        clean_m, clean_iv = [], []
        for m, iv in zip(m_vals, iv_vals):
            if abs(iv - mean) <= 2 * std:
                clean_m.append(m)
                clean_iv.append(iv)

        if len(clean_m) >= 10:
            try:
                coefs = np.polyfit(clean_m, clean_iv, deg=2)
                self.coefficients_by_chunk[chunk] = tuple(coefs[::-1])
            except:
                self.coefficients_by_chunk[chunk] = self.default_coeffs
        else:
            self.coefficients_by_chunk[chunk] = self.default_coeffs

    def get_smile_params(self) -> Optional[Tuple[float, float, float]]:
        return self.coefficients_by_chunk.get(self.current_chunk - 1)


    def save(self):
        return {
            "data_by_chunk": self.data_by_chunk,
            "coefficients_by_chunk": self.coefficients_by_chunk,
            "current_chunk": self.current_chunk
        }

    def load(self, obj):
        self.data_by_chunk = obj.get("data_by_chunk", {})
        self.coefficients_by_chunk = obj.get("coefficients_by_chunk", {})
        self.current_chunk = obj.get("current_chunk", 0)

# --- Helpers ---
#Define function
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm_pdf(d1) * np.sqrt(T)


def implied_volatility_hybrid(
    market_price, S, K, T, r=0.0, option_type="call", tol=1e-8, max_iter=100
):
    # --- 1단계: Bisection으로 rough guess
    low, high = 1e-6, 5.0
    for _ in range(50):
        mid = (low + high) / 2
        price = black_scholes_price(S, K, T, r, mid, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    sigma = (low + high) / 2  # 초기 추정치

    # --- 2단계: Newton-Raphson으로 refine
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma)
        if v < 1e-8:
            break
        sigma -= diff / v

    return float("nan")  # 실패시

def compute_T(current_ts: int, expiry_ts: int, timestamps_per_day: int = 100_000) -> float:
    remaining_ticks = expiry_ts - current_ts
    remaining_days = remaining_ticks / timestamps_per_day
    T = remaining_days / 252
    return T

def implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 20
) -> float:
    """Bisection method to find implied volatility"""
    low = 1e-6
    high = 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_price(S, K, T, r, mid, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2  # best estimate

def compute_bollinger(df: pd.DataFrame, col: str, window: int = 1000, k: float = 2.0):
    """
    df: DataFrame with a volatility time series column (e.g. 'iv')
    col: column name to calculate bands for
    window: rolling window size
    k: number of std devs from mean for bands
    """
    df["ma"] = df[col].rolling(window=window).mean()
    df["std"] = df[col].rolling(window=window).std()
    df["upper_band"] = df["ma"] + k * df["std"]
    df["lower_band"] = df["ma"] - k * df["std"]
    return df

# Voucher Strategy
class ImpliedVolStrategy:
    def __init__(self, option_symbol: str, config: dict, underlying_symbol: str):
        self.option = option_symbol  # ex: "VOLCANIC_ROCK_VOUCHER_10000"
        self.params = config         # PARAMS["VOLCANIC_ROCK_VOUCHER_10000"]
        self.volcanic_rock = underlying_symbol  # ex: "VOLCANIC_ROCK"

        # 전략 파라미터 초기화
        self.position_limit = self.params["position_limit"]
        self.window = self.params["iv_curve_fit_window"]
        self.iv_history: list[float] = []
        self.last_price = 0
        self.current_position = 0
        self.strike = self.params["strike"]
    
    def compute_vwap(self, order_dict: dict[int, int]) -> float:
        if not order_dict:
            return float('nan')
    
        total_volume = 0
        total_value = 0.0
        for price, volume in order_dict.items():
            if np.isnan(price): price = 0
            if np.isnan(volume): volume = 0
            abs_volume = abs(volume)  # volume은 bid면 양수, ask면 음수 → 절댓값 사용
            total_volume += abs_volume
            total_value += price * abs_volume
            if total_volume > 0:
                res = total_value / total_volume 
                self.last_price = res
            else:
                res = self.last_price
        return res
    
    def compute_implied_vol_from_vwap(
        self,
        state,
        expiry_timestamp: int = 999900 * 5,
        timestamps_per_day: int = 999900
    ) -> float:
        try:
            ts = state.timestamp
            T = compute_T(ts, expiry_timestamp, timestamps_per_day)
    
            voucher_depth = state.order_depths[self.option]
            underlying_depth = state.order_depths[self.volcanic_rock]
    
            S = self.compute_vwap(underlying_depth.buy_orders | underlying_depth.sell_orders)
            V = self.compute_vwap(voucher_depth.buy_orders | voucher_depth.sell_orders)
    
            if math.isnan(S) or math.isnan(V) or S <= 0 or V <= 0:
                return float('nan')
    
            iv = implied_volatility_hybrid(V, S, self.strike, T, r=0.0, option_type="call")
            
            logger.print(f"{self.option} VWAP | Underlying: {S}, Option: {V}")

            return iv
    
        except Exception as e:
            logger.print(f"[ERROR] IV calc failed: {e}")
            return float('nan')
    
    def update_iv(self, state):
        iv = self.compute_implied_vol_from_vwap(state)
        if not math.isnan(iv):
            self.iv_history.append(iv)
            if len(self.iv_history) > 200:
                self.iv_history.pop(0)  # 가장 오래된 값 제거
    
    def compute_iv_bollinger(self, window: int = 60, k: float = 2.0):
        if len(self.iv_history) < window:
            return None  # 데이터 부족
        df = pd.DataFrame({"iv": self.iv_history})
        df["ma"] = df["iv"].rolling(window=window).mean()
        df["std"] = df["iv"].rolling(window=window).std()
        df["upper_band"] = df["ma"] + k * df["std"]
        df["lower_band"] = df["ma"] - k * df["std"]
        return df.iloc[-1][["ma", "upper_band", "lower_band"]]
    
    
    def generate_iv_signal(self, current_iv: float) -> str:
        """
        Returns: "BUY", "SELL", "CLOSE", or "HOLD"
        """
        bands = self.compute_iv_bollinger()
        if bands is None:
            return "HOLD"  # 데이터 부족
    
        ma = bands["ma"]
        upper = bands["upper_band"]
        lower = bands["lower_band"]
    
        # 현재 포지션 없을 때 → 진입 조건
        if self.current_position == 0:
            if current_iv > upper:
                return "SELL"
            elif current_iv < lower:
                return "BUY"
            else:
                return "HOLD"
    
        # 이미 포지션 있는 경우 → 청산 조건
        elif self.current_position > 0:  # 롱 포지션 중 (BUY 진입함)
            if current_iv >= ma:
                return "CLOSE"
        elif self.current_position < 0:  # 숏 포지션 중 (SELL 진입함)
            if current_iv <= ma:
                return "CLOSE"
    
        return "HOLD"
    
   
    def run(
        self,
        state,
        position_limit: int = 200,
        lot_size: int = 10,
        aggressiveness: int = 1
    ) -> list[Order]:
        symbol = self.option  # 클래스 생성 시 지정된 옵션 이름
        iv = self.compute_implied_vol_from_vwap(state)
        self.update_iv(state)
    
        signal = "HOLD"
        if not math.isnan(iv):
            signal = self.generate_iv_signal(iv)
            logger.print(f"[{state.timestamp}] IV: {iv:.4f} | Signal: {signal}")
    
            if signal == "BUY":
                self.current_position = 1
            elif signal == "SELL":
                self.current_position = -1
            elif signal == "CLOSE":
                self.current_position = 0
    
        # 주문 생성 로직
        orders = []
        order_book = state.order_depths[symbol]
        position = state.position.get(symbol, 0)
    
        best_ask = min(order_book.sell_orders.keys()) if order_book.sell_orders else None
        best_bid = max(order_book.buy_orders.keys()) if order_book.buy_orders else None
    
        qty_to_trade = min(lot_size, position_limit - abs(position))
    
        if signal == "BUY" and best_ask is not None:
            price = best_ask + (1 if aggressiveness == 2 else 0)
            orders.append(Order(symbol, price, qty_to_trade))
    
        elif signal == "SELL" and best_bid is not None:
            price = best_bid - (1 if aggressiveness == 2 else 0)
            orders.append(Order(symbol, price, -qty_to_trade))
    
        elif signal == "CLOSE":
            if position > 0 and best_bid is not None:
                orders.append(Order(symbol, best_bid - aggressiveness, -position))
            elif position < 0 and best_ask is not None:
                orders.append(Order(symbol, best_ask + aggressiveness, -position))
        
        logger.print(f"[{state.timestamp}] {symbol} | IV: {iv:.4f} | Signal: {signal} | len(iv_history): {len(self.iv_history)}")
        logger.print(f"{symbol} Position: {position} | Qty to trade: {qty_to_trade}")
        logger.print(f"Orders: {orders}")
        return orders
    
    def save(self) -> dict:
        return {
            "iv_history": self.iv_history,
            "current_position": self.current_position,
        }

    def load(self, data: dict):
        self.iv_history = data.get("iv_history", [])
        self.current_position = data.get("current_position", 0)


class MacaronArbitrageStrategy:
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self.conversion_limit = params.get("conversion_limit", 10)
        self.storage_cost_per_unit = 0.1
        self.min_profit_margin = params.get("min_profit_margin", 3) # 3
        self.aggressive_post_offset = 1
        self.reversion_beta = params.get("reversion_beta", -0.5)  # -0.1

        # --- Mean reversion tracking ---
        self.buy_history = []
        self.sell_history = []
        self.history_limit = 1000 

    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        conversions = 0
        position = state.position.get(self.symbol, 0)

        max_buy_conversion = self.conversion_limit // 2
        max_sell_conversion = self.conversion_limit // 2

        order_depth = state.order_depths[self.symbol]
        best_bid_island = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_island = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        obs = state.observations.conversionObservations.get(self.symbol)
        if not obs:
            return orders, 0

        # Raw bakery values
        raw_bakery_buy = obs.askPrice + obs.transportFees + obs.importTariff
        raw_bakery_sell = obs.bidPrice - obs.transportFees - obs.exportTariff

        # Maintain rolling history
        self.buy_history.append(raw_bakery_buy)
        self.sell_history.append(raw_bakery_sell)

        if len(self.buy_history) > self.history_limit:
            self.buy_history.pop(0)
        if len(self.sell_history) > self.history_limit:
            self.sell_history.pop(0)

        # Calculate reversion-adjusted prices
        mean_buy = np.mean(self.buy_history)
        mean_sell = np.mean(self.sell_history)

        effective_bakery_buy = raw_bakery_buy + self.reversion_beta * (raw_bakery_buy - mean_buy)
        effective_bakery_sell = raw_bakery_sell + self.reversion_beta * (raw_bakery_sell - mean_sell)

        logger.print(f"[{state.timestamp}] Macaron Arbitrage | Bakery Buy: {effective_bakery_buy:.2f}, Bakery Sell: {effective_bakery_sell:.2f}")
        logger.print(f"Island | Best Bid: {best_bid_island}, Best Ask: {best_ask_island} | Current Position: {position}")

        # --- ACTIVE SELL POSTING STRATEGY ---
        if best_bid_island is not None:
            post_price = best_bid_island + self.aggressive_post_offset
            storage_cost = self.storage_cost_per_unit if position > 0 else 0
            adjusted_profit = post_price - effective_bakery_buy - storage_cost

            if adjusted_profit >= self.min_profit_margin:
                post_qty = max_sell_conversion
                orders.append(Order(self.symbol, post_price, -post_qty))
                conversions += post_qty
                logger.print(f" → POSTED SELL {post_qty} @ {post_price} (bakery buy: {effective_bakery_buy:.2f}) | Profit: {adjusted_profit:.2f}")

        #  --- ACTIVE BUY POSTING STRATEGY ---
        #if best_ask_island is not None:
        #    post_price = best_ask_island - self.aggressive_post_offset
        #    future_position_after_buy = position + max_buy_conversion
        #    storage_cost = self.storage_cost_per_unit if position > 0 or future_position_after_buy > 0 else 0
        #    adjusted_profit = effective_bakery_sell - post_price - storage_cost

        #    if adjusted_profit >= self.min_profit_margin:
        #        post_qty = max_buy_conversion
        #        orders.append(Order(self.symbol, post_price, post_qty))
        #        conversions += post_qty
        #        logger.print(f" → POSTED BUY {post_qty} @ {post_price} (bakery sell: {effective_bakery_sell:.2f}) | Profit: {adjusted_profit:.2f}")

        # --- BUY on ISLAND (then SELL to BAKERY) ---
        if best_ask_island is not None and effective_bakery_sell > best_ask_island:
            potential_buy_qty = min(abs(order_depth.sell_orders[best_ask_island]), max_buy_conversion - conversions)
            future_position_after_buy = position + potential_buy_qty
            storage_cost = self.storage_cost_per_unit if position > 0 or future_position_after_buy > 0 else 0
            adjusted_profit = effective_bakery_sell - best_ask_island - storage_cost

            if adjusted_profit >= self.min_profit_margin:
                buy_qty = min(potential_buy_qty, max_buy_conversion - conversions)
                if buy_qty > 0:
                    orders.append(Order(self.symbol, best_ask_island, buy_qty))
                    conversions += buy_qty
                    logger.print(f" → ARBITRAGE BUY {buy_qty} @ {best_ask_island}, sell to BAKERY @ {effective_bakery_sell:.2f}, Profit: {adjusted_profit:.2f}")

        # --- SELL on ISLAND (after BUYING from BAKERY) ---
        if best_bid_island is not None and best_bid_island > effective_bakery_buy:
            potential_sell_qty = min(abs(order_depth.buy_orders[best_bid_island]), max_sell_conversion - conversions)
            storage_cost = self.storage_cost_per_unit if position > 0 else 0
            adjusted_profit = best_bid_island - effective_bakery_buy - storage_cost

            if adjusted_profit >= self.min_profit_margin:
                sell_qty = min(potential_sell_qty, max_sell_conversion - conversions)
                if sell_qty > 0:
                    orders.append(Order(self.symbol, best_bid_island, -sell_qty))
                    conversions += sell_qty
                    logger.print(f" → ARBITRAGE SELL {sell_qty} @ {best_bid_island}, bought from BAKERY @ {effective_bakery_buy:.2f}, Profit: {adjusted_profit:.2f}")

        return orders, conversions

    def save(self):
        return {}

    def load(self, data):
        pass


# --- MAGNIFICENT MACARON STRATEGY ---
class MagnificentMacaronStrategy:
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self.cash = 0
        self.last_fair = None
        self.sunlight_history = []
        self.prev_slope = None  # Track previous slope for detecting trend reversal

    def get_fair_price(self, order_depth: OrderDepth, sunlightIndex: float) -> float:
        p = self.params
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
    
        if not bids or not asks:
            return self.last_fair if self.last_fair is not None else 0.0
    
        island_mid_price = (max(bids) + min(asks)) / 2
    
        # If sunlight is strong, just use mid price directly
        if sunlightIndex > p["CriticalSunlightIndex"]:
            fair = island_mid_price
        else:
            if self.last_fair is not None:
                fair = island_mid_price + p["reversion_beta"] * (island_mid_price - self.last_fair)
            else:
                fair = island_mid_price
    
        self.last_fair = fair
        return fair

    def run(self, state: TradingState) -> Tuple[List[Order]]:
        p = self.params
        obs = state.observations.conversionObservations[self.symbol]
        sunlightIndex = obs.sunlightIndex
        order_depth = state.order_depths[self.symbol]
    
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        orders = []
    
        max_buy = HARD_POSITION_LIMIT - position
        max_sell = HARD_POSITION_LIMIT + position
    
        # --- Fair price logic using sunlightIndex ---
        fair = self.get_fair_price(order_depth, sunlightIndex)
    
        # --- Sunlight slope tracking ---
        self.sunlight_history.append(sunlightIndex)
        if len(self.sunlight_history) > 10:
            self.sunlight_history.pop(0)
    
        slope = None
        if len(self.sunlight_history) >= 3:
            y = np.array(self.sunlight_history)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
    
        # --- BUY if sunlightIndex is low ---
        if self.prev_slope is not None and slope is not None:
            if sunlightIndex <= p["CriticalSunlightIndex"] and self.prev_slope < 0 and asks:
                best_ask, ask_vol = asks[0]
                buy_qty = min(-ask_vol, max_buy)
                if buy_qty > 0:
                    orders.append(Order(self.symbol, best_ask, buy_qty))
    
        # --- SELL if slope reverses from negative to positive ---
        if self.prev_slope is not None and slope is not None:
            if self.prev_slope <= 0 and slope > 0 and bids:
                best_bid, bid_vol = bids[0]
                sell_qty = min(bid_vol, max_sell)
                if sell_qty > 0:
                    orders.append(Order(self.symbol, best_bid, -sell_qty))
    
        self.prev_slope = slope  # update slope
    
        # --- TAKE orders ---
        for ask, vol in asks:
            if ask < fair - p["take_width"] and max_buy > 0:
                qty = min(-vol, max_buy)
                orders.append(Order(self.symbol, ask, qty))
                max_buy -= qty
    
        for bid, vol in bids:
            if bid > fair + p["take_width"] and max_sell > 0:
                qty = min(vol, max_sell)
                orders.append(Order(self.symbol, bid, -qty))
                max_sell -= qty
    
        # --- CLEAR orders ---
        position_after_take = position + (HARD_POSITION_LIMIT - max_buy) - (HARD_POSITION_LIMIT - max_sell)
        ask_clear = round(fair + p["clear_width"])
        bid_clear = round(fair - p["clear_width"])
    
        if position_after_take > 0:
            qty = min(position_after_take, max_sell)
            if qty > 0:
                orders.append(Order(self.symbol, ask_clear, -qty))
                max_sell -= qty
    
        if position_after_take < 0:
            qty = min(-position_after_take, max_buy)
            if qty > 0:
                orders.append(Order(self.symbol, bid_clear, qty))
                max_buy -= qty
    
        # --- MARKET MAKING ---
        bid_quote = round(fair - p["default_edge"])
        ask_quote = round(fair + p["default_edge"])
    
        if bids:
            best_bid = bids[0][0]
            if best_bid < fair - p["disregard_edge"]:
                bid_quote = best_bid + 1 if fair - best_bid > p["join_edge"] else best_bid
    
        if asks:
            best_ask = asks[0][0]
            if best_ask > fair + p["disregard_edge"]:
                ask_quote = best_ask - 1 if best_ask - fair > p["join_edge"] else best_ask
    
        # Position-based adjustment
        if position > p["soft_position_limit"]:
            ask_quote -= 1
        elif position < -p["soft_position_limit"]:
            bid_quote += 1
    
        mm_buy_qty = min(p["soft_position_limit"] - position, max_buy)
        mm_sell_qty = min(p["soft_position_limit"] + position, max_sell)
    
        if mm_buy_qty > 0:
            orders.append(Order(self.symbol, bid_quote, mm_buy_qty))
        if mm_sell_qty > 0:
            orders.append(Order(self.symbol, ask_quote, -mm_sell_qty))
    
        # --- PnL Tracking ---
        for trade in state.own_trades.get(self.symbol, []):
            self.cash += trade.quantity * (trade.price if trade.seller == "SUBMISSION" else -trade.price)
    
        mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else fair
        pnl = self.cash + position * mid
        logger.print(f"[{state.timestamp}] {self.symbol} | Fair: {fair:.2f} | Pos: {position} | PnL: {pnl:.2f}")
    
        return orders

    def save(self):
        return {"cash": self.cash, "last_fair": self.last_fair, "prev_slope": self.prev_slope}

    def load(self, data):
        self.cash = data.get("cash", 0)
        self.last_fair = data.get("last_fair", None)
        self.prev_slope = data.get("prev_slope", None)

# --- COPY OLIVIA STRATEGY ---
class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

class SignalStrategy(ABC):
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.signal = Signal.NEUTRAL
        self.orders: List[Order] = []
        self.conversions = 0

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        self.orders = []
        self.conversions = 0

        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths.get(self.symbol)

        if not order_depth:
            return self.orders, self.conversions

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_best_ask(order_depth), -position)
            elif position > 0:
                self.sell(self.get_best_bid(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_best_bid(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_best_ask(order_depth), self.limit - position)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def get_best_ask(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders)

    def get_best_bid(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders)

    def save(self) -> dict:
        return {"signal": self.signal.value}

    def load(self, data: dict) -> None:
        if data and "signal" in data:
            self.signal = Signal(data["signal"])


class OliviaCopySignalStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)

    def get_signal(self, state: TradingState) -> Signal | None:
        trades = state.market_trades.get(self.symbol, [])
        trades = [t for t in trades if t.timestamp == state.timestamp - 100]

        if any(t.buyer == "Olivia" for t in trades):
            return Signal.LONG
        elif any(t.seller == "Olivia" for t in trades):
            return Signal.SHORT

        return None

# --- TRADER ---
class Trader:
    def __init__(self):
        self.strategies = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", PARAMS["RAINFOREST_RESIN"]),
            #"SQUID_INK": SquidInkStrategy("SQUID_INK", PARAMS["SQUID_INK"]),
            "KELP": KelpStrategy("KELP", PARAMS["KELP"]),
            "PICNIC_BASKET1": Basket1Strategy("PICNIC_BASKET1", PARAMS["PICNIC_BASKET1"]),
            "PICNIC_BASKET2": Basket2Strategy("PICNIC_BASKET2", PARAMS["PICNIC_BASKET2"]),
            "VOLCANIC_ROCK_VOUCHER_10000": ImpliedVolStrategy("VOLCANIC_ROCK_VOUCHER_10000", PARAMS["VOLCANIC_ROCK_VOUCHER_10000"], "VOLCANIC_ROCK"),
            #"VOLCANIC_ROCK_VOUCHER_10250": ImpliedVolStrategy("VOLCANIC_ROCK_VOUCHER_10250", PARAMS["VOLCANIC_ROCK_VOUCHER_10250"], "VOLCANIC_ROCK"),
            #"VOLCANIC_ROCK_VOUCHER_10500": ImpliedVolStrategy("VOLCANIC_ROCK_VOUCHER_10500", PARAMS["VOLCANIC_ROCK_VOUCHER_10500"], "VOLCANIC_ROCK"),
            #"VOLCANIC_ROCK_VOUCHER_9750": ImpliedVolStrategy("VOLCANIC_ROCK_VOUCHER_9750", PARAMS["VOLCANIC_ROCK_VOUCHER_9750"], "VOLCANIC_ROCK"),
            #"VOLCANIC_ROCK_VOUCHER_9500": ImpliedVolStrategy("VOLCANIC_ROCK_VOUCHER_9500", PARAMS["VOLCANIC_ROCK_VOUCHER_9500"], "VOLCANIC_ROCK"),
            #"MACARON_ARBITRAGE": MacaronArbitrageStrategy("MAGNIFICENT_MACARONS", PARAMS["MAGNIFICENT_MACARONS"]),
            "MAGNIFICENT_MACARONS": MagnificentMacaronStrategy("MAGNIFICENT_MACARONS", PARAMS["MAGNIFICENT_MACARONS"])     
        }
        self.strategies["OLIVIA_COPY_SQUID_INK"] = OliviaCopySignalStrategy("SQUID_INK", HARD_POSITION_LIMIT)
        self.cross_arbitrage = CrossBasketArbitrage()

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        trader_data_in = jsonpickle.decode(state.traderData) if state.traderData else {}
        trader_data_out = {}
        result = {}
        conversions = 0

        for key, strategy in self.strategies.items():
            if key in trader_data_in:
                strategy.load(trader_data_in[key])

            symbol = getattr(strategy, "symbol", key)
            should_run = isinstance(strategy, SignalStrategy) or symbol in state.order_depths

            if should_run:
                try:
                    result_run = strategy.run(state)

                    # Normalize return type
                    if isinstance(result_run, list):
                        orders = result_run
                        conv = 0
                    else:
                        orders, conv = result_run

                    if isinstance(orders, Order):
                        orders = [orders]
                    elif not isinstance(orders, list):
                        logger.print(f"[WARN] Strategy {key} returned invalid orders: {orders}")
                        orders = []

                    result.setdefault(symbol, []).extend(orders)
                    conversions += conv

                except Exception as e:
                    logger.print(f"[ERROR] Strategy {key} failed: {e}")
                    continue

            trader_data_out[key] = strategy.save()
        
        # Add cross-basket orders
        cross_orders = self.cross_arbitrage.run(state)
        for symbol, order_list in cross_orders.items():
            if symbol in result:
                result[symbol].extend(order_list)
            else:
                result[symbol] = order_list

        # Run macaron arbitrage separately (uses conversions)
        arb_strategy = self.strategies.get("MACARON_ARBITRAGE")
        if arb_strategy:
            arb_orders, arb_conversions = arb_strategy.run(state)
            conversions += arb_conversions
            if "MAGNIFICENT_MACARONS" in result:
                result["MAGNIFICENT_MACARONS"].extend(arb_orders)
            else:
                result["MAGNIFICENT_MACARONS"] = arb_orders
        
        trader_data = jsonpickle.encode(trader_data_out)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data