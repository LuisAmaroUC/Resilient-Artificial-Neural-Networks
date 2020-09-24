#!/bin/bash
for (( i=$1 ; i<27 ; i++)) 							# 27 registers	
	do
	for ((j = $2; j< 64 ; j++))						# 64 bits in each one
		do 
			for ((k = $3; k< 5; k++))				# 5 bitflips are performed in each bit
			do 
				process_id=0
				process_id=`/bin/ps -fu $USER| grep ./testMnistDropout | grep -v "grep" | awk '{print $2}'`	# get the PID
				tag=$( tail -n 1 pathToFile/Results.txt )


				if [ $tag == "ACCURACY_VALUE" ]			# checks if the last result was a No Effect
				then 
					/ pathToFile/./pinject $process_id $i $j 300		
					echo "$i---$j" >> pathToFile/Results.txt  

					sleep 2

					
				elif [ $tag == "SEGMENTATIONFAULT" ] 		# checks if the last result was a CRASH
				then 
					pathToFile/./pinject $process_id $i $j 300	
					echo "$i---$j" >> pathToFile/Results.txt  

					sleep 3
				elif [ $tag == "$i---$j" ]			# checks if there was no new results
				then

					echo "HANG" >> pathToFile/Results.txt  
					((k=k-1))
					sleep 10
				elif [ $tag == "HANG" ]				# checks if the last result was a HANG
				then

					pathToFile/./pinject $process_id 0 0 300	

					((k=k-1))
					sleep 1

				else						
				
					pathToFile/./pinject $process_id $i $j 300	
					echo "$i---$j" >> pathToFile/Results.txt  

					sleep 2

				fi


		
			done
		done
	done


ps -ef | grep ./testMnistDropout | grep -v grep | awk '{print $2}' | xargs kill			# kills the process
