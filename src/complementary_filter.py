import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup
os.makedirs('figures', exist_ok=True)

df = pd.read_csv('motion.csv')
print(df.head())

df['t_s'] = (df['t_us'] - df['t_us'].iloc[0]) / 1e6

# Raw accelerometer 
plt.figure()
plt.plot(df['t_s'], df['ax'], label = 'ax')
plt.plot(df['t_s'], df['ay'], label = 'ay')
plt.plot(df['t_s'], df['az'], label = 'az')
plt.xlabel('Time (s)')
plt.ylabel('Accel (counts)')
plt.title('Raw accelerometer (motion)')
plt.legend()
plt.show()

# Raw gyroscope 
plt.figure()
plt.plot(df['t_s'], df['gx'], label = 'gx')
plt.plot(df['t_s'], df['gy'], label = 'gy')
plt.plot(df['t_s'], df['gz'], label = 'gz')
plt.xlabel('Time (s)')
plt.ylabel('Gyro (counts)')
plt.title('Raw gyroscope (motion)')
plt.legend()
plt.show()

# Gyro bias correction
gx_b = -551.687129
gy_b =  304.277888
gz_b =   59.532013

df['gx_c'] = df['gx'] - gx_b
df['gy_c'] = df['gy'] - gy_b
df['gz_c'] = df['gz'] - gz_b

GYRO_SF = 131.0  # counts per deg/s

df['gx_dps'] = df['gx_c'] / GYRO_SF
df['gy_dps'] = df['gy_c'] / GYRO_SF
df['gz_dps'] = df['gz_c'] / GYRO_SF

# Time step
t = df['t_s'].to_numpy()
dt = np.diff(t, prepend = t[0])
dt[0] = np.mean(dt[1:])

print('Mean dt:', np.mean(dt), 's')
print('Sampling rate:', 1 / np.mean(dt), 'Hz')

# Gyro-only roll 
df['roll_gyro'] = np.cumsum(df['gx_dps']) * np.mean(dt)

plt.figure()
plt.plot(df['t_s'], df['roll_gyro'], label = 'gyro-only')
plt.xlabel('Time (s)')
plt.ylabel('Roll angle (deg)')
plt.title('Gyroscope Integration (Roll) - Drift')
plt.legend()
plt.tight_layout()
plt.savefig('figures/gyro_roll_drift.png', dpi = 300)
plt.close()

# Accelerometer tilt
ACC_SF = 16384.0  # counts per g

df['ax_g'] = df['ax'] / ACC_SF
df['ay_g'] = df['ay'] / ACC_SF
df['az_g'] = df['az'] / ACC_SF

df['roll_acc'] = np.degrees(np.arctan2(df['ay_g'], df['az_g']))
df['pitch_acc'] = np.degrees(
    np.arctan2(-df['ax_g'], np.sqrt(df['ay_g'] ** 2 + df['az_g'] ** 2)))

# Accel-only roll 
plt.figure()
plt.plot(df['t_s'], df['roll_acc'], color = 'tab:orange', label = 'accel-only')
plt.xlabel('Time (s)')
plt.ylabel('Roll angle (deg)')
plt.title('Accelerometer Tilt Estimate (Roll)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/accel_roll_tilt.png', dpi = 300)
plt.close()

# Complementary filter
gyro = df['gx_dps'].to_numpy()
acc_angle = df['roll_acc'].to_numpy()

alpha = 0.98

roll_fused = np.zeros_like(acc_angle)
roll_fused[0] = acc_angle[0]

for k in range(1, len(roll_fused)):
    roll_pred = roll_fused[k - 1] + gyro[k] * dt[k]
    roll_fused[k] = alpha * roll_pred + (1 - alpha) * acc_angle[k]

df['roll_fused'] = roll_fused

# Fusion result 
plt.figure()
plt.plot(df['t_s'], df['roll_gyro'], label = 'gyro-only')
plt.plot(df['t_s'], df['roll_acc'], label = 'accel-only', alpha = 0.6)
plt.plot(df['t_s'], df['roll_fused'], label = 'fused')
plt.xlabel('Time (s)')
plt.ylabel('Roll angle (deg)')
plt.title(f'Complementary Filter (alpha = {alpha})')
plt.legend()
plt.tight_layout()
plt.savefig('figures/complementary_filter_roll.png', dpi = 300)
plt.close()
