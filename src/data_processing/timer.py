from time import perf_counter


class Timer:

    def __init__(self):
        self.timings = {}

    def start(self, section: str):
        self.timings[section] = {
            "start": perf_counter()
        }

    def stop(self, section: str):
        self.timings[section]["end"] = perf_counter()
        self.timings[section]["elapsed"] = (
            self.timings[section]["end"]
            - self.timings[section]["start"]
        )

    def report(self):

        print("\n" + "=" * 50)
        print("PERFORMANCE REPORT")
        print("=" * 50)

        total = 0

        for section, data in self.timings.items():

            elapsed = data["elapsed"]

            total += elapsed

            print(
                f"{section:<30}"
                f"{elapsed:>10.3f} sec"
            )

        print("-" * 50)
        print(
            f"{'TOTAL':<30}"
            f"{total:>10.3f} sec"
        )
        print("=" * 50)