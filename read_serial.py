
PORT = "COM3"      # replace with your real port from list_ports
BAUD = 115200      # must match firmware Serial.begin()

def file_lines(path):
    with open(path, "r") as f:
        for line in f:
            yield line
            
def serial_lines(PORT, BAUD):
    import serial
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        pass   
        
malformed = empty = valid = 0
source = file_lines("capture.txt")
with open("readings.csv", "w") as f:
    f.write("s0,s1,s2,s3,s4,s5\n")
    for line in source:
        line = line.strip()
        parts = line.split(',')
        if not line: 
            empty += 1
            continue
        if len(parts) != 6:
            malformed += 1
            continue
        try:
            values = [int(p) for p in parts]
        except ValueError:
            malformed += 1
            continue
        strValues = [str(v) for v in values]
        row = ",".join(strValues)
        valid += 1
        f.write(row + "\n")

