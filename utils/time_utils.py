import time


class Tic:
    def __init__(self):
        self.st = time.time()

    def tic(self):
        self.st = time.time()

    def tac(self):
        self.et = time.time()

    def print_time(self, process_name: str):
        self.tac()
        print(f"Process {process_name} takes {self.et - self.st}s.")
