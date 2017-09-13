import datetime
import time    
    
class Timer:
    def __init__(self, name, active=True):
        self.name = name if active else None

    def __enter__(self):
        self.start = time.time()
        self.last_tick = self.start
        return self

    def __exit__(self, *args):
        if self.name is not None:
            print("{} duration was {}.".format(self.name, self.readable(time.time() - self.start)))

    def get_time(self):
        return time.time() - self.start

    def readable(self, seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

    def tick(self, message):
        current = time.time()
        print("{} took {} ({} since last tick).".format(message, self.readable(current - self.start), self.readable(current - self.last_tick)))
        self.last_tick = current
        
