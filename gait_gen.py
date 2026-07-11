
def checksum(seq, timestamp, readings):
    return (seq + timestamp + sum(readings)) % 256

def make_frame(seq, timestamp, readings):
    ck = checksum(seq, timestamp, readings)
    
    readings_str = ",".join(map(str, readings))
    return f"INS,{seq},{timestamp},{readings_str},{ck}"

def parse_frame(line):
    parts = line.strip().split(",")

    sync = parts[0]
    if sync != "INS":
        return ("malformed", None)
    
    if len(parts) != 10:
        return ("malformed", None)
    
    field = parts[1:9]

    try:
        field = [int(f) for f in field]

        verified_sum = sum(field) % 256
        if verified_sum == int(parts[-1]):
            return ("ok", field)
        else:
            return ("bad_checksum", None)
    except ValueError:
        return ("malformed", None)
