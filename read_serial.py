import serial

PORT = "COM3"      # replace with your real port from list_ports
BAUD = 115200      # must match firmware Serial.begin()

try:
    with serial.Serial(PORT, BAUD, timeout=1) as ser, open("readings.csv", "w") as f:
        f.write("s0,s1,s2,s3,s4,s5\n")
        for i in range(20):
            raw = ser.readline()
            row = raw.decode().strip()
            if not row:
                continue
            print(row)
            f.write(row + "\n")
except serial.SerialException:
    print(f"Could not open {PORT}. Is the board plugged in and the port name correct?")