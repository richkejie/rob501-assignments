import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy

# new imports
import time
import matplotlib.pyplot as plt
import os
import shutil

# for emailing
import smtplib
from email.message import EmailMessage
import os

# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T

# change initial pose to a more challening one
C_init = dcm_from_rpy([np.pi/3,-np.pi/16,-np.pi/6])
t_init = np.array([[-0.05,8,-2]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

# set up dump folders
results_dir = "./results"
est_depths_dir = "./results/est_depths"
true_depths_dir = "./results/true_depths"

# clear out results dir
if os.path.isdir(results_dir):
    try:
        shutil.rmtree(results_dir)
        print(f"Directory '{results_dir}' and its contents cleared successfully.")
    except OSError as e:
        print(f"Error deleting directory '{results_dir}': {e}")
else:
    print(f"Directory '{results_dir}' does not exist.")

# create dirs
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Directory '{results_dir}' created.")
if not os.path.exists(est_depths_dir):
    os.makedirs(est_depths_dir)
    print(f"Directory '{est_depths_dir}' created.")
if not os.path.exists(true_depths_dir):
    os.makedirs(true_depths_dir)
    print(f"Directory '{true_depths_dir}' created.")

# Run sims --- estimate depths
start = 0.01
stop = 2
interval = 0.01
num = int((stop-start)/interval + 1)
gain_range = np.linspace(start=start,stop=stop,num=num)

print(f"\n\nTesting gains from {start} to {stop}, at intervals of {interval}.\n\n")

def main(dump_dir, do_depth, title, file_prefix):
    delta_t = np.zeros(len(gain_range))
    for i,gain in enumerate(gain_range):
        try:
            sim_start = time.time()
            ibvs_simulation(Twc_init,Twc_last,pts,K,gain,do_depth=do_depth)
            sim_end = time.time()
            delta_t[i] = sim_end - sim_start
            print(f"gain: {gain:.4}\tsim time: {delta_t[i]:.6}s")
        except Exception as e:
            delta_t[i] = np.inf
            print(f"gain: {gain:.4}\tcomputational error: {e}")

    plt.clf()
    plt.plot(gain_range,delta_t)
    plt.xlabel("Gain")
    plt.ylabel("Simulation Time (s)")
    plt.title(title)
    plt.savefig(f"{dump_dir}/{file_prefix}_ibvs_gain_v_time_plot.png")

    # find min time ---> best gain
    min_i = np.argmin(delta_t)
    best_time = delta_t[min_i]
    best_gain = gain_range[min_i]
    with open(f"{dump_dir}/{file_prefix}_best_gain.txt", 'w') as f:
        f.write(f"Best gain: {best_gain}\nConvergence time: {best_time}s\n")

    # save data
    with open(f"{dump_dir}/{file_prefix}_data.csv", 'w') as f:
        f.write("gain,time\n")
        for i in range(len(gain_range)):
            f.write(f"{gain_range[i]},{delta_t[i]}\n")

print("Running simulations for estimated depths...")
main(est_depths_dir, do_depth=True,
     title="Simulation Convergence Time vs Gain - Estimated Depths",
     file_prefix="est")
print("\n\n")

print("Running simulations for true depths...")
main(true_depths_dir, do_depth=False,
     title="Simulation Convergence Time vs Gain - True Depths",
     file_prefix="true")
print("\n\n")

#------------------------------------------------------#
#------------------------------------------------------#
#------------------------------------------------------#

# --- Email details ---
sender_email = "***"
receiver_email = "***"
app_password = "***"

with open("email.txt", 'r') as f:
    text = f.read().splitlines()
    sender_email = text[0]
    receiver_email = text[0]
    app_password = text[1]

subject = "ROB501 A4 Part 4 Simulation Results"
body = "See attached files for results.\n"

# --- Create the email ---
msg = EmailMessage()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = subject
msg.set_content(body)

# --- Attach files ---
for filename in os.listdir(est_depths_dir):
    filepath = os.path.join(est_depths_dir, filename)
    if os.path.isfile(filepath):  # Ignore subfolders
        with open(filepath, "rb") as f:
            file_data = f.read()
            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="octet-stream",
                filename=filename
            )

for filename in os.listdir(true_depths_dir):
    filepath = os.path.join(true_depths_dir, filename)
    if os.path.isfile(filepath):  # Ignore subfolders
        with open(filepath, "rb") as f:
            file_data = f.read()
            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="octet-stream",
                filename=filename
            )

# --- Send via SMTP ---
# For Gmail, you’ll need an App Password (not your normal password)
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(sender_email, app_password)
    smtp.send_message(msg)

print("Email sent successfully!")


