from datetime import datetime

def get_timestamp() -> str:
    current_time = datetime.now()
    return "".join([str(current_time.year), str(current_time.month), str(current_time.day), str(current_time.hour), str(current_time.minute)])