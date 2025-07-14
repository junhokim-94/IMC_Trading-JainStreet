# IMC Trading Competition: Prosperity3    
# Team JainStreet Strategy Overview


This repository hosts the trading bot that secured 9th place in the manual‑trading section and 138th place overall in the IMC Prosperity 3 competition. Each strategy resides in its own class and the `Trader` class decides which combination to deploy in real time. The outline below captures the essential logic of every module without exposing proprietary parameter values.

## ResinStrategy (RAINFOREST_RESIN)
A pure market‑making engine for RAINFOREST_RESIN. It computes a short‑term fair value from recent quotes and submits or removes liquidity whenever the bid–ask spread drifts outside a predefined band. Inventory is bounded by soft and hard limits.

## SquidInkStrategy (SQUID_INK)
A relative‑value strategy between SQUID_INK and RAINFOREST_RESIN. A linear regression of long‑run price series defines the theoretical spread. The live Z‑score of that spread drives entries and exits, favouring mean reversion.

## KelpStrategy (KELP)
Targets micro‑structure noise in KELP. Large dubious lots are filtered out before estimating a micro mid‑price. A negative beta term encourages mean reversion, and orders are staged in three layers—Take, Clear, and Market‑Make—to balance execution cost and liquidity provision.

## Basket1Strategy (PICNIC_BASKET1)
Tracks the spread between PICNIC_BASKET1 and a weighted synthetic basket of CROISSANTS, JAMS, and DJEMBES. When the Z‑score of the spread breaches a statistical band the strategy buys or sells the basket expecting convergence.

## Basket2Strategy (PICNIC_BASKET2)
Mirrors Basket1Strategy but focuses on PICNIC_BASKET2 against a different weight mix of CROISSANTS and JAMS, with its own inventory caps.

## CrossBasketArbitrage
Defines a cross spread: two units of PICNIC_BASKET1 versus three units of PICNIC_BASKET2 plus two units of DJEMBES. Divergence beyond a threshold triggers simultaneous trades across the three legs to lock in the imbalance.

## ImpliedVolStrategy (VOLCANIC_ROCK_VOUCHER)
Derives implied volatility from the volume‑weighted average price of VOLCANIC_ROCK_VOUCHER options and applies a rolling Bollinger‑style band. Volatility above the upper band opens short positions, below the lower band opens long positions, and reversion to the mean closes them.

## MagnificentMacaronStrategy (MAGNIFICENT_MACARONS)
Estimates fair value by blending Island exchange quotes and Bakery over‑the‑counter prices. When the sunlight index exceeds a threshold it trusts the simple mid‑price; otherwise it adds a mean‑reversion component. Changes in the slope of the index are treated as momentum signals.

## MacaronArbitrageStrategy (MAGNIFICENT_MACARONS, Conversion)
Executes risk‑free conversions between Island and Bakery when price gaps cover storage costs and a minimum profit margin. The engine automatically selects the cheaper venue to buy and the dearer venue to sell, then performs the conversion to crystallise the spread.

## OliviaCopySignalStrategy (SQUID_INK)
Observes the most recent tick trades from a rival trader named Olivia. A buy by Olivia is treated as a transient long signal, and a sell as a transient short signal. In the absence of fresh signals, the position is gradually unwound.

## Logger and Risk Management
The `Logger` compresses each round’s state and orders into JSON for rapid post‑mortem analysis. All modules respect a global `HARD_POSITION_LIMIT` in addition to their local `position_limit` parameters.

---
For full implementation details and hyper‑parameters, consult the source code inside each strategy class.

