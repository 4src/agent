# vim: set ts=2:sw=2:et:
import re
import sys
import math
import random
import traceback
from copy import deepcopy
from termcolor import colored
from ast import literal_eval 

isa  = isinstance
r    = random.random
seed = random.seed

def shuffle(a:list) ->list: random.shuffle(a); return a 

class O(object):
  n=0
  def __init__(o1, **d): o1.__dict__.update(**o1.slots(**d));  O.n+=1; o1._id=O.n
  def slots(o1,**d)    : return d
  def __repr__(o1)     : return o1.__class__.__name__+"{"+(" ".join(
                                 sorted([f":{k} {v}" for k, v in o1.__dict__.items() 
                                         if k[0] != '_'])))+"}"
  def __hash__(o1)     : return o1._id
  def update(o1,d)     :
    for k,v in  d.__dict__.items():  o1.__dict__[k] = v

def coerce(str):
  try: return literal_eval(str)
  except: return str

def settings(help:str,update=False) -> O:
  "Parses help string for lines with flags (on left) and defaults (on right)"
  d={}
  for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",help):
    k,v = m[1], m[2]
    d[k] = coerce( cli(k,v) if update else v )
  return O(**d)

def cli(k:str, v:str) -> str:
  """If there exists a command-line flag `-k[0]`, then update `v`.
  For non-booleans, take value from command-line.
  For booleans, just flip the default."""
  for i,x in enumerate(sys.argv):
    if ("-"+k[0]) == x:
      v="False" if v=="True" else ("True" if v=="False" else sys.argv[i+1])
  return v

def csv(file:str):
  "Iterator for CSV files"
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
         yield [coerce(cell.strip()) for cell in line.split(",")]

def yell(s,c):
  "print string `s` in bold text, in color `c`"
  print(colored(s,c,attrs=["bold"]),end="")

def tests(the,funs):
  """if asked (using -g str), run some `fun` in `funs` (or run all if `-g all`).
  After each run, reset seed and `the` to  the defaults. If `fun` crashes, 
  increment `fails` and move on to next demo. Return number of fails."""
  cache = deepcopy(the) # always reset `the` to `cache` 
  tries,fails = 0,0
  for fun in funs:
    k = fun.__name__
    if the.go==k or the.go=="all":
      yell(k+" ","light_yellow")
      ok = True
      try:
        the.update(cache)
        seed(the.seed)
        tries += 1
        if fun()==False: ok=False
      except:
        print(traceback.format_exc())
        ok=False
      if ok: yell("PASS\n","light_green")
      else:  yell("FAIL\n","light_red"); fails += 1
  if the.go != "nothing":
    yell(O(tries=tries, fails=fails),"magenta"); print("")
  return fails

def main(the,help,funs):
  the.update(settings(help, update=True))
  yell(help,"yellow") if the.help else sys.exit(tests(the,funs))
