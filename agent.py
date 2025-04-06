# vim: set ts=2:sw=2:et:
"""
agent.py: multi-objective analysis and planning   
(c) 2023 Tim Menzies <timm@ieee.org> BSD-2   

USAGE:  
    python3 tests.py [OPTIONS] [-g ACTION]

OPTIONS:   

    -b  --bins    initial number of bins      = 16  
    -B  --Beam    explore the top 'Beam' vals = 10
    -c  --cohen   'same' if under sd*cohen    = .35  
    -d  --digits  digits for stats reports    = 2   
    -f  --file    where to get data           = ../data/auto93.csv  
    -g  --go      start up action             = nothing   
    -G  --Goal    plan,fear,dare,seek         = plan  
    -m  --min     stop at len**min            = .5    
    -h  --help    show help                   = False  
    -r  --rest    how many other to select    = 3
    -s  --seed    random number seed          = 1234567891   

"""
from functools import cmp_to_key
import itertools
from lib import *

the = settings(__doc__)

# --------------------------------------------------------------------
def COLS(a: list[str]) -> O:
  """Factory for making COLS things. Applied to first line of csv file.
  Uppercase words become NUMs (and others are SYMs). Dependent `y`
  goal attributes are marked as a klass (with a trailing `!`) or
  as something to minimize or maximize (with a training `-` or `+`.
  Columns to ignore end in `X`."""
  cols = O(all=[], x=[], y=[], klass=None, names=a)
  for c, s in enumerate(a):
    w = -1 if s[-1] == "-" else 1
    col = NUM(at=c, txt=s, w=w) if s[0].isupper() else SYM(at=c, txt=s)
    cols.all += [col]
    if s[-1] != "X":
      (cols.y if s[-1] in "!+-" else cols.x).append(col)
      if s[-1] == "!":
        cols.klass = col
  return cols

# -----------------------------/---------------------------------------
class COL(O):
  "Defines a class that lets type hints to mention NUMs or SYMs"
  pass

class SYM(COL):
  """Summarize a stream of non-numbers."""
  def slots(self, at=0, txt=" "):
    return dict(at=at, txt=txt, n=0, most=0, mode=None, has={})

class NUM(COL):
  """Summarize a stream of numbers."""
  def slots(self, at=0, txt=" ", w=1):
    return dict(at=at, txt=txt, n=0, w=w, lo=inf, hi=-inf, mu=0, m2=0)

def add(col: COL, x, inc=1):
  "add `x` to `col`; repeat `inc` number of times"
  if x != "?":
    col.n += inc
    if isa(col, SYM):
      now = col.has[x] = col.has.get(x, 0) + inc
      if now > col.most:
        col.most, col.mode = now, x
    else:
      for _ in range(inc):
        d = x - col.mu
        col.mu += d / col.n
        col.m2 += d * (x - col.mu)
        col.lo = min(x, col.lo)
        col.hi = max(x, col.hi)
  return x

def norm(num: NUM, n):
  "n normalizes 0..1 for min..max"
  return n if n == "?" else (n - num.lo) / (num.hi - num.lo + 1/inf)

def mid(col: COL):
  "central tendency of an existing col (mean or mode)"
  return col.mu if isa(col, NUM) else col.mode

def div(col: COL) -> float:
  "diversity of an existing col (standard deviation or entropy)"
  def e(p): return p * math.log(p, 2)
  return (
    -sum(e(n / col.n) for n in col.has.values())
    if isa(col, SYM)
    else (col.m2 / (col.n - 1)) ** 0.5
  )

def merged(sym1: SYM, sym2: SYM) -> SYM:
  "if merging simplifies things, return that merge"
  sym12 = deepcopy(sym1)
  for x, n in sym2.has.items():
    add(sym12, x, n)
  if div(sym12) <= (sym1.n * div(sym1) + sym2.n * div(sym2)) / sym12.n:
    return sym12

# --------------------------------------------------------------------
class ROW(O):
  """ROWs store cells and, optionally, may have some label. Discretized
  ROWs store the descrete values in `cooked`."""
  def slots(self, cells=[]):
    return dict(cells=cells, label=None, cooked=[])

def DATA(src, rows=[]) -> O:
  """Factor from making data things from either files,
  or mimicking the structure of another data"""
  data = O(rows=[], cols=None)
  if type(src) == str:
    [adds(data, x) for x in csv(src)]
  elif type(src) == O:
    adds(data, src.cols.names)
  [adds(data, row) for row in rows]
  return data

def adds(data: O, x):
  "update `data` with one row `x`. if this is top row, create headers."
  if not data.cols:
    data.cols = COLS(x)
  else:
    row = x if type(x) == ROW else ROW(cells=x)
    data.rows += [row]
    for cols in [data.cols.x, data.cols.y]:
      for col in cols:
        add(col, row.cells[col.at])

def stats(data: DATA, cols: list[COL] = None, fun=mid):
  "summarize `cols`"
  def rnd(x): return round(x, ndigits=the.digits) if isa(x, float) else x
  out = {col.txt: rnd(fun(col)) for col in (cols or data.cols.y)}
  out["N"] = len(data.rows)
  return O(**out)

def better(data: DATA, row1: ROW, row2: ROW) -> bool:
  "true if `row1` should be ranked ahead of `row2`"
  s1, s2, cols, n = 0, 0, data.cols.y, len(data.cols.y)
  for col in cols:
    a, b = norm(col, row1.cells[col.at]), norm(col, row2.cells[col.at])
    s1 -= math.exp(col.w * (a - b) / n)
    s2 -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n

def betters(data: DATA, rows: list[ROW] = None) -> list[ROW]:
  def fun(r1, r2): return better(data, r1, r2)
  return sorted(rows or data.rows, key=cmp_to_key(fun))

# --------------------------------------------------------------------
class BIN(O):
  """Store and update a set of `rows`, their min and max values (for
  one column) as well as a count of the labels seen in those rows"""
  def slots(self, at=0, txt="", xlo=None, xhi=None, rows=None, labels=None):
    return dict(
      at=at,
      txt=txt,
      xlo=xlo or 0,
      xhi=xhi or xlo,
      rows=rows or set(),
      labels=labels or SYM(),
      score=0)

def binned(bin, x, row):
  if x != "?":
    bin.xlo = min(x, bin.xlo)
    bin.xhi = max(x, bin.xhi)
    add(bin.labels, row.label)
    bin.rows.add(row)

def what(b,r,B,R):
  b,r,z  = b/B, r/R, 10**-64
  return dict(
    plan = lambda: b**2/(b+r+z),
    fear = lambda: r**2/(b+r+z),
    dare = lambda: (b+r)/(abs(b-r)+z),
    seek = lambda: 1/(b+r+z))[the.Goal]()

def orAnd(bins,B,R):
  a={}
  for  b in sorted(bins,key=lambda b:b.at):
    a[b.at] = (b.rows.union(a[b.at])) if b.at in a else b.rows
  out=SYM()
  a = set.intersection(*map(set, a.values())) 
  for row in a:
    add(out, row.label)
  return O(bins=bins, rows=a,
           score=what(out.has.get(True,0), out.has.get(False,0),B,R)) 

def bins(rows: list[ROW], at: int, txt: str, xcol: COL, **d):
  xfun = lambda row: row.cells[xcol.at]
  if isa(xcol, NUM):
    return numBins(rows=rows, at=at, txt=txt, xfun=xfun, **d)
  else:
    tmp = {k: BIN(at=at, txt=txt, xlo=k) for k in xcol.has.keys()}
    [binned(tmp[xfun(row)], xfun(row), row) for row in rows]
    return sorted(tmp.values(), key=lambda bin: bin.xlo)

def numBins(rows: list[ROW], at: int, txt: str, xfun: callable, 
            eps: float, small: float) -> list[BIN]:
  rows = sorted((r for r in rows if xfun(r) != "?"), key=xfun)
  bin = BIN(at=at, txt=txt, xlo=xfun(rows[0]))
  out = []
  for i, row in enumerate(rows):
    xhere = xfun(row)
    binned(bin, xhere, row)
    if bin.xhi - bin.xlo > eps:
      if len(bin.rows) > small and i < len(rows) - small:
        if xhere != xfun(rows[i + 1]):
          out += [bin]
          bin = BIN(at=at, txt=txt, xlo=xhere)
  if len(bin.rows) > 0:
    out += [bin]
  return merges(out)

def merges(bins: list[BIN]) -> list[BIN]:
  tmp, i = [], 0
  while i < len(bins):
    a = bins[i]
    if i < len(bins) - 1:
      b = bins[i + 1]
      if labels := merged(a.labels, b.labels):
        a = BIN(
          at=a.at,
          txt=a.txt,
          xlo=a.xlo,
          xhi=b.xhi,
          rows=a.rows | b.rows,
          labels=labels)
        i = i + 1
    tmp += [a]
    i = i + 1
  return bins if len(tmp) == len(bins) else merges(tmp)
# --------------------------------------------------------------------
def rankBins(data, best,rest):
  lst=[]
  for col in data.cols.x:
    for bin in bins(best+rest, at=col.at, txt=col.txt, xcol=col,
                    eps=div(col)*the.cohen, small=col.n**the.min):
      bin.score = what(bin.labels.has.get(True,0),
                       bin.labels.has.get(False,0),
                       len(best), len(rest))
      lst += [bin]
  return sorted(lst, key=lambda b:b.score)[-min(the.Beam, len(lst)):]

def rankRules(bins, best,rest):
  [print(">",len(b.rows)) for b in bins]
  subsets = [[]]
  for x in bins:
    subsets += [s+[x] for s in subsets]
  out =[]
  for sub in subsets[1:]:
    out += [orAnd(sub,len(best),len(rest))]
  return sorted(out,key=lambda x:x.score)
  #for subs in subsets[:1]: print(subs)


