import threading
import time

import paramiko
import re

# List of machines (replace with actual machine IPs or hostnames)
machines = [
    'dsinlp01',
    'dgx01',
    'dgx02',
    'dgx03',
]

# Jump node login details
jump_node = 'dsihead.lnx.biu.ac.il'
jump_user = 'aeirew'
jump_pass = 'ban7J6zM'  # Alternatively, you can use SSH keys.

check_interval = 60


# Function to check GPU utilization
def check_gpu_utilization(machine, client):
    stdin, stdout, stderr = client.exec_command(f'ssh {machine} nvidia-smi')
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    if error:
        print(f"An error occurred while checking GPUs on {machine}: {error}")
    return output


# Function to highlight free or not fully utilized GPUs
def highlight_free_gpus(output):
    # Regular expression to capture GPU utilization percentage
    pattern = re.compile(r'(\d+)%')
    free_gpus = []

    lines = output.splitlines()
    for line in lines:
        match = pattern.search(line)
        if match:
            utilization = int(match.group(1))
            if utilization < 10:  # less than 10% means "not fully utilized"
                free_gpus.append(line)

    return free_gpus


# Function to send notification
def notify_user(machine, free_gpus):
    gpu_details = "\n".join(free_gpus)
    notification_message = f"Free GPUs found on {machine}:\n{gpu_details}"
    print(notification_message)  # Prints to console for logging

    # Notify the user via system notification
    # os.system(f'notify-send "Free GPU Alert" "{notification_message}"')


# Connect to the jump node and check GPUs on all machines
def check_all_gpus_in_background():
    jump_client = None
    try:
        # Create an SSH client
        jump_client = paramiko.SSHClient()
        jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the jump node
        jump_client.connect(jump_node, username=jump_user, password=jump_pass)

        while True:
            for machine in machines:
                print(f"Checking {machine}...")
                gpu_output = check_gpu_utilization(machine, jump_client)
                free_gpus = highlight_free_gpus(gpu_output)

                if free_gpus:
                    notify_user(machine, free_gpus)
                else:
                    print(f"All GPUs on {machine} are busy.")

            time.sleep(check_interval)  # Wait for the specified interval before checking again

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the SSH connection
        if jump_client:
            jump_client.close()


if __name__ == "__main__":
    background_thread = threading.Thread(target=check_all_gpus_in_background, daemon=True)
    background_thread.start()

    # Keep the main script alive
    try:
        while background_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping the GPU monitoring script.")
