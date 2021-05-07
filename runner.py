import os
import subprocess
import socket


def syslog(message, host, level=6, facility=1,  port=514):
    """
    Send syslog UDP packet to given host and port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = '<%d> %s' % (level + facility*8, message)
    sock.sendto(bytes(data, "utf-8"), (host, port))
    sock.close()


def run_with_logging(command, address):
    my_env = os.environ.copy()
    my_env["WANDB_PROJECT"] = "Test Bengali Run"

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True, env=my_env)

    while True:
        output = proc.stdout.readline().rstrip()
        if proc.poll() is not None:
            break
        if output:
            if output[0] != '[' or "verag" in output:
                print(output)
            syslog(output, host=address)
