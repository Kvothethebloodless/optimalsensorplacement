import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ax.set_xticks()
# ax.set_yticks()


# READ DATA FROM FILE

road_filename = os.path.join('simulationdata/','road.npy')
roaddata = np.load(road_filename)
road_points = roaddata
target_loc = roaddata[1]


timedata_filename = os.path.join('simulationresults/','psodata.npy')
timeseriesdata = np.load(timedata_filename)
alldata = timeseriesdata
pos_data = alldata[0]
centroid_data = alldata[1]
spread_data = alldata[2]


no_frames = np.size(spread_data)
print('111111111111111\n')
print no_frames
print('111111111111111\n')


no_particles = np.shape(pos_data[0])[0]


# $no_particles = no_particles;
# return[pos_data,centroid_data,spread_data,no_frames]


colors = np.random.random(no_particles)

# min_pos_x = -55565656
# max_pos_x = -min_pos_x
#
# max_pos_y = max_pos_x
# min_pos_y = min_pos_x
#
#



min_pos_x =np.min(np.min(pos_data[0]))-5
min_pos_y = np.min(np.min(pos_data[1]))-5
max_pos_x = np.max(np.max(pos_data[0]))+5
max_pos_y = np.max(np.max(pos_data[1]))+5


#min_pos_x = -10
#min_pos_y = 10
#max_pos_x = -100
#kmax_pos_y = 10

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(15, 15))
# ax = fig.add_axes([0, 0, 1, 1], frameon=True)
ax = fig.gca()
ax.set_xlim(min_pos_x, max_pos_x), ax.set_xticks(np.arange(min_pos_x, max_pos_x, .5))

ax.set_ylim(min_pos_y, max_pos_y), ax.set_yticks(np.arange(min_pos_x, max_pos_x, .5))
ax.set_autoscale_on
# plt.grid()
# ax.plot(road_points[0], road_points[1])



scat = ax.scatter(pos_data[0][:, 0], pos_data[0][:, 1], lw=.05, s=20, c=colors)
ax.grid(True)


# def readdata():


# def main(packed_data):
# [pos_data,centroid_data,spread_data,no_frames]=packed_data;

# numframes = no_frames;
# numpoints = np.shape(pos_data[1]);
# numpoints = numpoints[0];
# color_data = np.random.random((numframes, numpoints))
# x, y, c = np.random.random((3, numpoints))

# fig = plt.figure()
# scat = plt.scatter(pos_data[0][:,0], pos_data[0][:,1], c=c, s=100)



fno = np.array([])


def update_plot(frame_number):
    if np.sum(fno == frame_number):
        print('--')
        name = str(frame_number) + '.png'
        plt.savefig(name, bbox_inches='tight')
    else:
        print('00')
    print(frame_number)
    scat.set_offsets(pos_data[frame_number])


animation = FuncAnimation(fig, update_plot, frames=no_frames, interval=100, repeat=False)

plt.show()
fig.clear()