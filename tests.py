import sys
from agent import  __doc__ as help
from agent import *
from lib import *

funs=[]
def go(f): global funs; funs += [f]; return f

@go
def thed(): print(the)

@go 
def csvd():
  n=0
  for a in csv(the.file): n += len(a)
  return n==3192

@go
def lohid():
  num = NUM()
  [add(num,x) for x in range(20)]
  return 0==num.lo and 19==num.hi

@go
def numd():
  num = NUM()
  [add(num,r()) for x in range(10**4)]
  return .28 < div(num) < .32 and .49 < mid(num) < .51

@go
def symd():
  sym = SYM()
  [add(sym,c) for c in "aaaabbc"]
  return 1.37 < div(sym) < 1.38 and mid(sym)=='a'

@go
def statd():
  d0  = DATA(the.file)
  s0  = stats(d0); a0 = s0.__dict__['Acc+']
  return  15.5 < a0 < 15.6

@go
def betterd():
  d0 = DATA(the.file)
  d1 = DATA(d0, rows=betters(d0)[-30:])
  d2 = DATA(d0, rows=betters(d0)[:-30])
  s0 = stats(d0); a0 = s0.__dict__['Acc+']
  s1 = stats(d1); a1 = s1.__dict__['Acc+']
  s2 = stats(d2); a2 = s2.__dict__['Acc+']
  print(a2,a0,a1)
  return a2 < a0  < a1

@go
def mimikd():
  d0 = DATA(the.file)
  d1 = DATA(d0, rows=d0.rows)
  return d0.cols.y[1].m2 == d1.cols.y[1].m2

@go
def binsd():
  n=30
  d0 = DATA(the.file)
  rows = betters(d0)
  best = rows[-n:]
  rest = shuffle(rows[:-n])[:n*the.rest]
  d1 = DATA(d0, rows=best)
  d2 = DATA(d0, rows=rest)
  print("\nall ",stats(d0))
  print("best",stats(d1))
  print("rest",stats(d2))
  for r in  best: r.label=True
  for r in  rest: r.label=False
  for b in rankBins(d0,best,rest):
      print(b.score, b.txt, b.xlo,b.xhi,b.labels.has)
  for x in rankRules(rankBins(d0,best,rest),best,rest):
    print(x.score,len(x.bins))
  # for col in  d0.cols.x:
  #   print("")
  #   for b in bins(best+rest, at=col.at, txt=col.txt, xcol=col,
  #                        eps=the.cohen*div(col),  small=col.n**the.min):
  #     print("!!",isa(col,NUM), b.txt, b.xlo,b.xhi,b.labels.has, len(b.rows))
  #
if __name__ == "__main__":
  main(the, help, funs)
