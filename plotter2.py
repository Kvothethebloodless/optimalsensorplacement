import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


#ax.set_xticks()
#ax.set_yticks()


#READ DATA FROM FILE
timeseriesdata = np.load('psodata2.npy');
alldata = timeseriesdata;
pos_data = alldata[0];
centroid_data = alldata[1];
spread_data = alldata[2];
no_frames = np.size(spread_data);
no_particles = np.shape(pos_data[0])[0];
#$no_particles = no_particles;
#return[pos_data,centroid_data,spread_data,no_frames]
colors = np.random.random(no_particles)


min_pos_x = -55565656;
max_pos_x = -min_pos_x;

max_pos_y = max_pos_x;
min_pos_y = min_pos_x;


#for n in range(no_frames):
	#print(n)
	#temp_max_pos_x = np.max(pos_data[n][:,0]);
	#temp_min_pos_x = np.min(pos_data[n][:,0]);

	#temp_max_pos_y = np.max(pos_data[n][:,1]);
	#temp_min_pos_y = np.min(pos_data[n][:,1]);
	#a = temp_max_pos_x>max_pos_x;
	#max_pos_x = a*temp_max_pos_x + int(not(a))*max_pos_x;
	#a = temp_max_pos_y > max_pos_y
	#max_pos_y = a*temp_max_pos_y + int(not(a))*max_pos_y;
	#
	#a = temp_min_pos_y < min_pos_y
	#min_pos_y = a*temp_min_pos_y + int(not(a))*min_pos_y;
	#a = temp_min_pos_x > min_pos_x
	#min_pos_x = a*temp_min_pos_x + int(not(a))*min_pos_x;
	#




min_pos_x = -15
min_pos_y = -15
max_pos_x = 15
max_pos_y =15;

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0,0,1,1], frameon=True)
ax.set_xlim(min_pos_x,max_pos_x), ax.set_xticks(np.arange(min_pos_x,max_pos_x,1))
ax.set_ylim(min_pos_y,max_pos_y), ax.set_yticks(np.arange(min_pos_x,max_pos_x,1))

# Create rain data
#rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
#                                      ('size',     float, 1),
#                                      ('growth',   float, 1),
#                                      ('color',    float, 4)])

# Initialize the raindrops in random positions and with
# random growth rates.
#rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
#rain_drops['growth'] = np.random.uniform(50, 200, n_drops)

# Construct the scatter which we will update during animation
# as the raindrops develop.

scat = ax.scatter(pos_data[0][:, 0], pos_data[0][:, 1],lw=1,s=200, c=colors);
#ax.grid(True)


#def readdata():


#def main(packed_data):
	#[pos_data,centroid_data,spread_data,no_frames]=packed_data;

	#numframes = no_frames;
	#numpoints = np.shape(pos_data[1]);
	#numpoints = numpoints[0];
	#color_data = np.random.random((numframes, numpoints))
	#x, y, c = np.random.random((3, numpoints))

	#fig = plt.figure()
	#scat = plt.scatter(pos_data[0][:,0], pos_data[0][:,1], c=c, s=100)



fno = np.array([])
def update_plot(frame_number):

	if np.sum(fno==frame_number):
		print('--')
		name = str(frame_number)+'.png'
		plt.savefig(name,bbox_inches='tight')
	else:
		print('00')
	print(frame_number)
	scat.set_offsets(pos_data[frame_number])


animation = FuncAnimation(fig, update_plot,frames =no_frames, interval=100,repeat=False)

plt.show()