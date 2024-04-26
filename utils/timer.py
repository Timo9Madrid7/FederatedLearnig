from datetime import datetime
from typing import Tuple
import time

def get_timestamp() -> str:
    current_time = datetime.now()
    return "".join([str(current_time.year), str(current_time.month), str(current_time.day), str(current_time.hour), str(current_time.minute)])

class Timer:
    start_time = 0
    end_time = 0
    def start(self) -> None:
        self.start_time = time.time()
    
    def end(self) -> None:
        self.end_time = time.time()
    
    def timeslot(self) -> Tuple[int, int]:
        elapsed_time = self.end_time - self.start_time
        hours, minutes, seconds = int(elapsed_time // 3600), int(elapsed_time // 60), int(elapsed_time % 60)
        return hours, minutes, seconds
    

if __name__ == '__main__':
    timer = Timer()

    timer.start()
    time.sleep(2.2)
    timer.end()

    print(timer.timeslot())    