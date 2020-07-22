#!/bin/bash
for (( i=$1 ; i<27 ; i++))	
	do
	for ((j = $2; j< 64 ; j++))
		do 
			for ((k = $3; k< 5; k++))
			do 
				process_id=0
				process_id=`/bin/ps -fu $USER| grep ./testMnist | grep -v "grep" | awk '{print $2}'`

				tag=$( tail -n 1 /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/Dropout0.txt )

				


				echo $process_id

				if [ $tag == "0.8831" ]
				then 
					/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject $process_id $i $j 300	
					echo "$i---$j" >> /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/Dropout0.txt

					echo $i $j

					sleep 2

					
				elif [ $tag == "SEGMENTATIONFAULT" ] 
				then 
					/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject $process_id $i $j 300	
					echo "$i---$j" >> /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/Dropout0.txt

					echo $i $j

					sleep 3
				elif [ $tag == "$i---$j" ]
				then
					echo "EQUAL VALUES"
					echo $tag

					echo "HANG" >> /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/Dropout0.txt
					((k=k-1))
					sleep 10
				elif [ $tag == "HANG" ]
				then
					echo "HANG"
					echo $tag

					/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject $process_id 0 0 300	

					((k=k-1))
					sleep 1

				else
				
					/home/luisamaro/Desktop/HW_Injectors_ucXception/newest_injector/./pinject $process_id $i $j 300	
					echo "$i---$j" >> /home/luisamaro/Desktop/examples-master/cpp/mnist/build/ResultsStimDrop50/Dropout0.txt

					echo $i $j
					echo $tag

					sleep 2

				fi


		
			done
		done
	done


ps -ef | grep ./testMnistDropout | grep -v grep | awk '{print $2}' | xargs kill