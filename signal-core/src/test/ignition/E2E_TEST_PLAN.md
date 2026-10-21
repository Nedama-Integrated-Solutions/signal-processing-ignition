# End-to-end test plan

The random tests prove each function against SciPy. These phases prove the whole system: the signed module on a
real gateway, in every scope, on real plant data, under concurrent load. Run every phase on 8.1, then on 8.3.

## Phase 1: install and lifecycle

1. Install the release `.modl`. Config, Modules shows Signal Processing 0.1.0 as Running, with no licence needed.
2. Check the gateway logs (Status, Logs): no errors or warnings from the module.
3. Restart the gateway. The module comes back Running, and `system.signal.butter(4, 0.2)` works in the Script
   Console.
4. Install the same file again over itself. It installs cleanly and keeps working.
5. Uninstall. `system.signal` is gone (calls fail with AttributeError). Reinstall: it works again.

Pass: no errors in the logs at any step, and the functions are available after each install.

## Phase 2: every scope

1. Designer: the accuracy test (`random_check.py --export`, then `target/accuracy_test.py`). 64 PASS checks,
   then `compare_ignition.py` ends in PASS.
2. Gateway: the spot check (`gateway_check.py`, steps at its top). 34 rows end in PASS.
3. Gateway timer: a timer script every 10 s that filters the last 10 minutes of a tag with `sosfiltfilt` and
   writes the last filtered value to a memory tag. Leave it running for an hour.
4. Perspective: a view with a chart of raw and filtered history, the filtered series computed in a script
   transform with `butter` and `sosfiltfilt` (Perspective scripts run in gateway scope).
5. Vision client, if Vision is installed: a button whose script prints `system.signal.medfilt([1, 5, 2, 9, 3])`.

Pass: correct results in every scope; the timer's memory tag tracks the filtered signal for the whole hour with no
errors in the logs.

## Phase 3: real plant data

1. In `e2e_realdata.py`, set `TAGS` to 3 to 5 real tags of different kinds: a noisy flow, a slow temperature, a
   level with spikes, and a quantised or integer tag. Set `HOURS` (6 to 24) and `INTERVAL_SECONDS`.
2. Run it in the Designer Script Console, then
   `python src/test/python/compare_realdata.py --results <path it prints>`.
3. Optional: run it in gateway scope too, from a project library script and a message handler, as for the spot
   check.

Pass: every result within tolerance of SciPy (compare_realdata.py ends in PASS), and the timings it records are
reasonable for the data size.

## Phase 4: concurrency and load

1. Run `e2e_concurrency.py` in the Designer Script Console (8 threads, 25 rounds of 10 calls).
2. Run it in gateway scope, from a project library script and a message handler.
3. Raise `THREADS` to 16 and `ROUNDS` to 100 and run it in gateway scope while watching Status, Performance (CPU
   and memory).

Pass: zero mismatches and zero errors every time; memory returns to its baseline afterwards.

## Phase 5: failure behaviour

1. Historian gaps not filled: pass a list with None to `sosfiltfilt`. A ValueError explains the problem; nothing
   is logged as a Java stack trace.
2. Wrong arguments from a script (a string for `N`, a negative `nperseg`, an unknown keyword): ValueError or
   TypeError, never a Java exception.
3. A large input: `sosfiltfilt` on 10,000,000 points in gateway scope. It completes; the gateway stays healthy.

Pass: errors are clear Python exceptions, and the gateway is unaffected.
