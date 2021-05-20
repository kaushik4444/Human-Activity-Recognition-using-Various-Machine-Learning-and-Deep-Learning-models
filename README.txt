Basketball Game Activity Recognition - RAML SoSe 21

Data overview:
	- 5+1 classes (shooting, layups, dribbling, running, walking and null) of basketball specific movements 
	- two subjects
	- sensors:
		- eSense (left ear @50Hz)
		- Bangle.js (right wrist (dominant hand) @92Hz ±8g)
		- Bangle.js (right ankle @92Hz ±8g)
	- files
		- wrist + ankle: one (or two) csv-file(s) per activity and subject 
			- 
		- earbud: one csv-file per subject
			- NOTE: subject 2's file contains null class; still there are null parts within subject 1's file
			- HINT: identify the timestamps at which each activity within the videos was conducted and relabel the files accordingly

Folders:
	- sensor_data: collected data from each sensor per each subject
	- videos: GoPro videos of the recorded activites (one version with timestamp (ts); one without)  

Conditions: 
	- partly cloudy, afternoon

Video: 
	- GoPro Hero5 (1080p @60FPS, linear FOV (except dribbling, walking and running for subject 2 -> 'superview' = extreme wide FOV))

General notes/ hints:
	- the timestamp within the sensor files is incorrect; only the start timestamp is correct; use the Hz information to correctly modify the timestamp within each file
	- the labels are not correct; use the timestamp information obtained by looking at the videos to identify when each activity started/ stopped
	- sensors were started/ stopped at different times; you need to identify overlapping timeframes for which you have data from all sensors
	- the sampling rate of the wrist-worn sensors is different from the ear-worn sensor -> keyword: upsampling/ downsampling!
	 