results = open('./test/results', 'r')


# counters
level_tp = 0
level_tn = 0
level_fn = 0
level_fp = 0
zebra_tp = 0
zebra_tn = 0
zebra_fn = 0
zebra_fp = 0
videos = 200


for result in results:
	file_name = result.split(" ")[0]
	query_level = int(list(result.split(" ")[1])[0])
	query_zebra = int(list(result.split(" ")[1])[1])
	solutions = open('/home/bobz/repos/OpenCV-programs/visionhack/AI/Hack/trainset/train_test.txt', 'r')
	for solution in solutions:
		if file_name in solution:
			true_level = int(list(solution.split(" ")[1])[0])
			true_zebra = int(list(solution.split(" ")[1])[1])
			if(true_zebra - query_zebra == -1):
				zebra_fp +=1
			elif(true_zebra - query_zebra == 1):
				zebra_fn +=1
			if(true_level - query_level == -1):
				level_fp +=1
			elif(true_level - query_level == 1):
				level_fn +=1
	solutions.close()

TOTAL_ZEBRA = 53
TOTAL_LEVEL = 40
zebra_tp = TOTAL_ZEBRA - zebra_fn
zebra_tn = videos - (zebra_tn+zebra_fn+zebra_fp)
level_tp = TOTAL_LEVEL - level_fn
level_tn = videos - (level_tn+level_fn+level_fp)
total_zebra = (-1000*zebra_fp+100*zebra_tp)/TOTAL_ZEBRA
total_level = (-1000*level_fp+100*level_tp)/TOTAL_LEVEL


print "zebra: tp:"+str(zebra_tp)+" tn:"+str(zebra_tn)+" fn:"+str(zebra_fn)+" fp:"+str(zebra_fp)+" total:"+str(total_zebra)
print "level: tp:"+str(level_tp)+" tn:"+str(level_tn)+" fn:"+str(level_fn)+" fp:"+str(level_fp)+" total:"+str(total_level)
results.close()

