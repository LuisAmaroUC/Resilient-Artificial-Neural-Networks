#!/bin/bash

#After an error has occurred the program restarts automatically


tag=$( tail -n 1 pathToFile/Results.txt) 		#Reads the last line of a .txt file


while (! ./testMnistDropout)  
	do
	
	echo "SEGMENTATIONFAULT" >> pathToFile/Results.txt		#writes SEGMENTATIONFAULT in the file

	
	echo "Restarting program..."	
		
	./script.sh
		
done


