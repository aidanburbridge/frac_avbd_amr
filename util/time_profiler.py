
from time import perf_counter
from contextlib import contextmanager
from collections import defaultdict

class PhaseProfiler:
    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    @contextmanager
    def phase(self, name:str):
        s = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - s
            self.t[name] += dt
            self.n[name] += 1
    
    def report(self, total_name = "frame"):
        lines = []
        total = sum(self.t.values()) or 1e-12
        for k, v in sorted(self.t.items(), key=lambda kv:kv[1], reverse=True):
            c = self.n[k]
            lines.append(f"\t{k:20s}\t{v*1000:8.2f} ms\t({v/total*100:5.1f}%)\tn={c}\tavg={v/c*1000:7.3f} ms")
        return "PROFILE SUMMARY\n" + "\n".join(lines) + "\n"
    
    def reset(self):
        self.t.clear()
        self.n.clear()