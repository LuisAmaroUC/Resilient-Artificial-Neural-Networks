#!/bin/bash

tag=$( tail -n 1 /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/Dropout0.txt)


if (./testMnist)
	then 
		
		if [ $tag == "HANG" ]
		then 
			
			./srcipt.sh
		fi
	else
		echo "SEGMENTATIONFAULT" >> /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/Dropout0.txt

	
		echo "Restarting program..."	
		
		./script.sh
		
fi	

