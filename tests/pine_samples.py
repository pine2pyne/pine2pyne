"""Small Pine scripts the tests convert. Each one exercises a few language features of its version."""

INDICATOR_V6 = '''//@version=6
indicator("My MA", overlay=true)
length = input.int(20, "Length")
src = input.source(close, "Source")
ma = ta.sma(src, length)
plot(ma, "MA", color=color.blue)
'''

STRATEGY_V6 = '''//@version=6
strategy("Crossing", overlay=true, initial_capital=10000)
fastLen = input.int(5, "Fast")
slowLen = input.int(20, "Slow")
fast = ta.sma(close, fastLen)
slow = ta.sma(close, slowLen)
if ta.crossover(fast, slow)
    strategy.entry("L", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("L")
plot(fast, "Fast", color=color.teal)
plot(slow, "Slow", color=color.orange)
'''

# a user type, an enum, a method, a typed array, `for ... in`, `var`, history reads
TYPES_V6 = '''//@version=6
indicator("Levels", overlay=true)
enum Side
    up
    down
type Level
    float price
    Side side
    int bars = 0
method age(Level this) =>
    this.bars += 1
    this.bars
var levels = array.new<Level>()
if not na(ta.pivothigh(high, 5, 5))
    levels.push(Level.new(high[5], Side.up))
if not na(ta.pivotlow(low, 5, 5))
    levels.push(Level.new(low[5], Side.down))
oldest = 0
for lvl in levels
    oldest := math.max(oldest, lvl.age())
plot(levels.size())
plot(oldest)
'''

# `var`, a `switch` used as a value, `plotshape`, a history read of a user variable
INDICATOR_V5 = '''//@version=5
indicator("Momentum", overlay=false)
length = input.int(14, "Length", minval=1)
src = input.source(close, "Source")
var float peak = na
mom = src - src[length]
peak := na(peak) ? mom : math.max(peak, mom)
state = switch
    mom > 0 => 1
    mom < 0 => -1
    => 0
plot(mom, "Momentum")
plot(peak, "Peak")
plotshape(state == 1 and state[1] != 1, style=shape.triangleup)
'''

# `study`, unprefixed functions, a typed `input(...)`, `transp`, a colour-or-na ternary
STUDY_V4 = '''//@version=4
study("Bands", overlay=true)
len = input(20, "Length", type=input.integer)
mult = input(2.0, "Mult")
basis = sma(close, len)
dev = mult * stdev(close, len)
plot(basis, color=color.blue, transp=50)
plot(basis + dev)
plot(basis - dev)
bgcolor(close > basis ? color.green : na, transp=90)
'''

LIBRARY_V6 = '''//@version=6
library("MathLib")
export clamp(float x, float lo, float hi) =>
    math.max(lo, math.min(hi, x))
export method half(float x) =>
    x / 2
'''

USES_LIBRARY_V6 = '''//@version=6
indicator("Uses a library")
import someone/MathLib/1 as ml
plot(ml.clamp(close, low, high))
'''

# TradingView refuses an assignment to a history read
REFUSED_HISTORY_TARGET = '''//@version=6
indicator("Bad")
x = close
x[1] := 2
plot(x)
'''

# a `ta.*` call made only on some bars: converted, with a warning
CONDITIONAL_TA_CALL = '''//@version=6
indicator("Warn")
v = 0.0
if close > open
    v := ta.sma(close, 10)
plot(v)
'''

COMMENT_ONLY = '''// nothing but comments
// and blank lines

'''

SYNTAX_ERROR = '''//@version=6
indicator("Broken")
plot(close +
'''
