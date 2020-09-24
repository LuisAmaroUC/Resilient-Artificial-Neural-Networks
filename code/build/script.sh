#!/bin/bash

#After an error has occurred the program restarts automatically


tag=$( tail -n 1 /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/reg4.txt ) 		#Reads the last line of a .txt file


while (! ./testMnist)  
	do
	
	echo "SEGMENTATIONFAULT" >>  /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/reg4.txt		#writes SEGMENTATIONFAULT in the file

	
	echo "Restarting program..."	
		
	./script.sh
		
done
