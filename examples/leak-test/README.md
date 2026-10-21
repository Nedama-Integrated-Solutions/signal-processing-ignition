# Example: pressure rate-of-rise leak test

A leak test as run on vacuum chambers, freeze dryers, autoclaves and sealed vessels: pump the chamber down,
isolate it, let it settle, then measure how fast the pressure rises. Leak rate = volume x rise rate (mbar.L/s).
`system.signal.detrend` does the analysis:

- a linear detrend gives the fitted line, so the rise rate, and the residual noise;
- a constant detrend gives the total variation, so R squared;
- a detrend with breakpoints at the quarter points gives the rise rate at the start and at the end of the test. A
  rate that is still falling means outgassing (gas released from surfaces), not a leak, and the test reports
  INCONCLUSIVE instead of a false FAIL.

A simulator runs one test every 14 minutes on a 500 L chamber, each with a new true leak rate, so every result can
be checked against the truth. Point the application at a real chamber when it has proven itself.

## Setup (Ignition 8.1 or 8.3, with a tag history provider)

1. **Project library script.** In the Designer, Scripting, Project Library: create a script named `leaktest` and
   paste `leaktest.py` into it. Save the project.
2. **Tags.** In the Script Console, with your history provider's name:
   ```python
   print leaktest.setup("YourHistoryProvider")
   ```
   This creates `[default]LeakTest` with Config, Sim, State and Results folders. In the Tag Browser, open
   `LeakTest/Sim/Pressure` and check its History section: enabled, your provider, Deadband Style Discrete,
   Deadband 0 (every sample is stored).
3. **Simulator.** Gateway Events, Timer: add a timer named `LeakTestSim`, fixed rate, 1000 ms, with the script:
   ```python
   leaktest.simulate_step()
   ```
4. **Test edges.** Gateway Events, Tag Change: add a script named `LeakTestEdges` on the tag path
   `[default]LeakTest/Sim/TestActive`, with the script:
   ```python
   if not initialChange and previousValue is not None:
       if newValue.value and not previousValue.value:
           system.tag.writeBlocking(["[default]LeakTest/State/TestStart"], [newValue.timestamp])
       elif previousValue.value and not newValue.value:
           start = system.tag.readBlocking(["[default]LeakTest/State/TestStart"])[0].value
           leaktest.evaluate_later(start, newValue.timestamp)
   ```
   It records the start when the test begins and, when it ends, waits 15 s for the historian to store the last
   samples, then evaluates.
5. **Save the project.** The simulator starts at once; the first test ends after about 11 minutes.
6. **Perspective view** (optional): labels bound to `LeakTest/Results/Verdict`, `LeakRate`, `Message`,
   `RateRatio` and `R2`; a Power Chart with a pen on `LeakTest/Sim/Pressure`; a Table bound to
   `LeakTest/Results/Log`, which lists every test with the true and measured leak rates.

## Checking it

After two hours (8 or 9 tests), in the Script Console:
```python
log = system.tag.readBlocking(["[default]LeakTest/Results/Log"])[0].value
for r in range(log.rowCount):
    print [log.getValueAt(r, c) for c in range(log.columnCount)]
```

Pass: every `verdict` equals `expected`, and every `error_sigmas` (measured minus true, in standard uncertainties)
is within plus or minus 4. Offline, 240 simulated tests gave the right verdict every time, with the worst error at
3.5 standard uncertainties.

To see the outgassing check work, set `LeakTest/Config/SettleSeconds` to 10: the next tests report INCONCLUSIVE.
Set it back to 180 afterwards.

## Using it on a real chamber

Set `PRESSURE` at the top of `leaktest.py` to the chamber's pressure tag (historised with deadband 0), set
`LeakTest/Sim/Enabled` to false, point the tag change script at the PLC's test-active signal, and set
`Config/Volume_L`, `Config/Limit` and `Config/SettleSeconds` for the chamber.
