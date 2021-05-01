import subprocess
import socket


def syslog(message, host, level=6, facility=1,  port=514):
    """
    Send syslog UDP packet to given host and port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = '<%d> %s' % (level + facility*8, message)
    sock.sendto(bytes(data, "ascii"), (host, port))
    sock.close()


def run_with_logging(command, address):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)

    while True:
        output = proc.stdout.readline().rstrip()
        if proc.poll() is not None:
            break
        if output:
            print(output)
            syslog(output, host=address)
