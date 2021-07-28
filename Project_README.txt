Basketball Game Activity Recognition - RAML SoSe 21

Data overview:
	- 5+1 classes (shooting, layups, dribbling, running, walking and null) of basketball specific movements 
	- two subjects
	- sensors:
		- eSense (left ear @50Hz ±8g)
		- Bangle.js (right wrist (dominant hand) @92Hz ±8g)
		- Bangle.js (right ankle @92Hz ±8g)
	- files
		- wrist + ankle: one (or two) csv-file(s) per activity and subject 
			- columns: timestamp, acc_x, acc_y, acc_z
		- earbud: one csv-file per subject
			- columns: timestamp, device_name, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, label

Folders:
	- sensor_data: collected data from each sensor per each subject
	- videos: GoPro videos of the recorded activites (one version with timestamp (ts); one without)  

Conditions: 
	- partly cloudy, afternoon

Video: 
	- GoPro Hero5 (1080p @60FPS, linear FOV (except dribbling, walking and running for subject 2 -> 'superview' = extreme wide FOV))

First steps + general hints:
	1. Correct the timestamp within the Bangle.js files - as the first timestamp is correct within each file and you know at which rate was sampled, you can caculate the correct timestamp 
	   for each record.
	2. Upsample or downsample either the Bangle.js or eSense files using a method of your choice so that all files are assuming the same sampling rate (Hz).
	3. Watch the videos and write down the timestamps at which each activity happenend for which subject.
	4. Merge the all three files into one dataframe (Hint: you need to account for there not being records of all three sensors at all time since sensors were started/ stopped at 
	   different times).
	5. Apply the labeling obtained in step 3 to the dataframe you obtained (remember that there is also a null class!)
	6. Apply deep learning/ ML
	 
