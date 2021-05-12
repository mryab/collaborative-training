import os
import subprocess
import socket


def login_with_wandb():
    """ TEMPORARY JANKINESS WARNING: this code will set up error logging to our wandb account. """
    with open(f'{os.environ["HOME"]}/.netrc', 'w') as f:
        f.write("""
machine api.wandb.ai
  login user
  password 7cc938e45e63ef7d2f88f811be240ba0395c02dd
""")



def syslog(message, host, level=6, facility=1,  port=514):
    """
    Send syslog UDP packet to given host and port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = '<%d> %s' % (level + facility*8, message)
    sock.sendto(bytes(data, "utf-8"), (host, port))
    sock.close()


def run_with_logging(command, address, wandb_login: bool = False):
    my_env = os.environ.copy()
    my_env["WANDB_ENTITY"] = "learning-at-home"
    my_env["WANDB_PROJECT"] = "Worker_logs"
    if wandb_login:
        login_with_wandb()

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True, env=my_env)

    while True:
        output = proc.stdout.readline().rstrip()
        if proc.poll() is not None:
            break
        if output:
            if output[0] != '[' or  "__main__" in output or "verag" in output:
                print(output)
            syslog(output, host=address)
