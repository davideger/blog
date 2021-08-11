import math
import resource
import time
import sys


def clock2():
  """clock2() -> (t_user,t_system)

  Similar to clock(), but return a tuple of user/system times.
  """
  return resource.getrusage(resource.RUSAGE_SELF)[:2]


def _format_time(timespan, precision=3):
  """Formats the timespan in a human readable form"""
  if timespan >= 60.0:
    # we have more than a minute, format that in a human readable form
    # Idea from http://snipplr.com/view/5713/
    parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
    time_parts = []
    leftover = timespan
    for suffix, length in parts:
      value = int(leftover / length)
      if value > 0:
        leftover = leftover % length
        time_parts.append(u"%s%s" % (str(value), suffix))
      if leftover < 1:
        break
    return " ".join(time_parts)
  # Unfortunately the unicode 'micro' symbol can cause problems in
  # certain terminals.
  # See bug: https://bugs.launchpad.net/ipython/+bug/348466
  # Try to prevent crashes by being more secure than it needs to
  # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
  units = [u"s", u"ms", u"us", "ns"]  # the save value
  if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
    try:
      u"\xb5".encode(sys.stdout.encoding)
      units = [u"s", u"ms", u"\xb5s", "ns"]
    except:
      pass
  scaling = [1, 1e3, 1e6, 1e9]
  if timespan > 0.0:
    order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
  else:
    order = 3
  return u"%.*g %s" % (precision, timespan * scaling[order], units[order])


def timeit(code, glob=globals(), local_ns=locals()):
  wtime = time.time
  # time execution
  wall_st = wtime()
  st = clock2()
  out = eval(code, glob, local_ns)
  end = clock2()
  wall_end = wtime()
  # Compute actual times and report
  wall_time = wall_end - wall_st
  cpu_user = end[0] - st[0]
  cpu_sys = end[1] - st[1]
  cpu_tot = cpu_user + cpu_sys
  # On windows cpu_sys is always zero, so no new information to the next print
  if sys.platform != "win32":
    print("CPU times: user %s, sys: %s, total: %s" % \
         (_format_time(cpu_user),_format_time(cpu_sys),_format_time(cpu_tot)))
  print("Wall time: %s" % _format_time(wall_time))
  return out
