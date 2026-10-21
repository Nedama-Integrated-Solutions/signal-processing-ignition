"""
Source checks that keep the Ignition layer in step with the core. random_check.py runs these every time.

    1. Every core function has a method in SignalScriptModule (8.1 and 8.3) whose @KeywordArgs names match
       the core signature in order, and which passes its own name to JythonBridge.call.
    2. The 8.1 and 8.3 copies are identical apart from the package names and the 8.3 annotation
       (JythonElement instead of ScriptFunction).
    3. The script hints file has .desc, .returns and one .param entry per parameter, nothing else, in ASCII.
"""

import difflib
import re

METHOD = re.compile(
    r'@KeywordArgs\(\s*names\s*=\s*\{(?P<names>[^}]*)\}.*?\)\s*'
    r'public\s+PyObject\s+(?P<method>\w+)\s*\(\s*PyObject\[\]\s+\w+\s*,\s*String\[\]\s+\w+\s*\)\s*\{\s*'
    r'return\s+JythonBridge\.call\(\s*"(?P<call>\w+)"', re.S)
PUBLIC_METHOD = re.compile(r'public\s+PyObject\s+\w+\s*\(')
SCOPES = ("common", "gateway", "client", "designer")


def java_dir(repo, scope, v):
    return repo / ("%s-%s" % (scope, v)) / "src" / "main" / "java" / "com" / "nedama" / "signal" / ("%s%s" % (scope, v))


def hints_path(repo, v):
    return (repo / ("common-%s" % v) / "src" / "main" / "resources" / "com" / "nedama" / "signal" / ("common%s" % v)
            / "SignalScriptModule.properties")


def normalise(text):
    text = re.sub(r"\b(common|gateway|client|designer)8[13]\b", r"\1XX", text)
    return text.replace("ScriptFunction", "JythonElement")


def check(repo, signatures):
    """Returns [(check name, [problems])]. An empty problem list means the check passed."""
    return [
        ("script module 8.1", script_module(repo, "81", signatures)),
        ("script module 8.3", script_module(repo, "83", signatures)),
        ("8.1 and 8.3 sources match", mirror(repo)),
        ("script hints", hints(repo, signatures)),
    ]


def script_module(repo, v, signatures):
    text = (java_dir(repo, "common", v) / "SignalScriptModule.java").read_text(encoding="utf-8")
    found = {}
    for m in METHOD.finditer(text):
        names = [n.strip().strip('"') for n in m.group("names").split(",") if n.strip()]
        found[m.group("method")] = (names, m.group("call"))
    problems = []
    if len(PUBLIC_METHOD.findall(text)) != len(found):
        problems.append("a public method is missing @KeywordArgs or does not call JythonBridge.call directly")
    for fn, sig in signatures.items():
        if fn not in found:
            problems.append("%s: no script method" % fn)
            continue
        names, call = found[fn]
        if names != sig["params"]:
            problems.append("%s: @KeywordArgs names %s, core signature %s" % (fn, names, sig["params"]))
        if call != fn:
            problems.append("%s: calls JythonBridge.call(\"%s\")" % (fn, call))
    for fn in found:
        if fn not in signatures:
            problems.append("%s: script method with no core function" % fn)
    return problems


def mirror(repo):
    problems = []
    for scope in SCOPES:
        d81, d83 = java_dir(repo, scope, "81"), java_dir(repo, scope, "83")
        names = sorted({p.name for p in d81.glob("*.java")} | {p.name for p in d83.glob("*.java")})
        for name in names:
            a, b = d81 / name, d83 / name
            if not a.exists() or not b.exists():
                problems.append("%s/%s exists for only one version" % (scope, name))
                continue
            first = first_difference(normalise(a.read_text(encoding="utf-8")), normalise(b.read_text(encoding="utf-8")))
            if first:
                problems.append("%s/%s differs: %s" % (scope, name, first))
    a, b = hints_path(repo, "81").read_text(encoding="utf-8"), hints_path(repo, "83").read_text(encoding="utf-8")
    first = first_difference(a, b)
    if first:
        problems.append("SignalScriptModule.properties differs: %s" % first)
    return problems


def first_difference(a, b):
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if line.startswith(("-", "+")) and not line.startswith(("---", "+++")):
            return line.strip()
    return None


def hints(repo, signatures):
    raw = hints_path(repo, "81").read_bytes()
    problems = []
    try:
        raw.decode("ascii")
    except UnicodeDecodeError:
        problems.append("the hints file must be ASCII (write other characters as \\uXXXX)")
    keys = set()
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, sep, value = line.partition("=")
            if not sep or not value.strip():
                problems.append("empty or malformed entry: %s" % line)
            keys.add(key.strip())
    expected = set()
    for fn, sig in signatures.items():
        expected |= {fn + ".desc", fn + ".returns"} | {"%s.param.%s" % (fn, p) for p in sig["params"]}
    problems += ["missing hint %s" % k for k in sorted(expected - keys)]
    problems += ["hint for nothing: %s" % k for k in sorted(keys - expected)]
    return problems
