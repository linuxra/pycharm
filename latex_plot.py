import matplotlib.pyplot as plt
import numpy as np

# Create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y, label='sin(x)')

# Add a title
ax.set_title('Example of Text and Annotations in Matplotlib')

# Add an x-label
ax.set_xlabel('x-axis')

# Add a y-label
ax.set_ylabel('y-axis')

# Add a legend
ax.legend()

# Add text
ax.text(5, 0.5, 'Text Example', fontsize=12, color='red', backgroundcolor='yellow')

# Add a mathematical expression using LaTeX
ax.text(1, -0.75, r'$y = \sin(x)$', fontsize=16, color='blue')

# Annotate a specific point
point_x = np.pi
point_y = np.sin(point_x)
ax.plot(point_x, point_y, marker='o', color='green', markersize=8)
ax.annotate("Annotated Point\n($\pi$, sin($\pi$))", xy=(point_x, point_y), xytext=(point_x + 1, point_y + 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to generate the sine wave data
def sine_wave(t, amplitude=1, frequency=1, phase=0):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

# Initialization function for the animation
def init():
    line.set_data([], [])
    return line,

# Update function for the animation
def update(frame):
    t = np.linspace(0, 2 * np.pi, 1000)
    y = sine_wave(t, amplitude=1, frequency=1, phase=-frame / 10)
    line.set_data(t, y)
    return line,

# Create a plot
fig, ax = plt.subplots()
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1, 1)

# Create an empty line to be updated
line, = ax.plot([], [], lw=2)

# Create the animation using FuncAnimation
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=50)

# To display the animation in Jupyter Notebook or other environments, uncomment the following line:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# Save the animation as a GIF (requires ImageMagick)
# ani.save('sine_wave_animation.gif', writer='imagemagick', fps=20)

# Show the plot (if running as a standalone script)
plt.show()
