from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
from datetime import datetime,timedelta
import datetime as dt_time
import calendar 
import pytz
from rupture_demo import segment_packets

HUB_IP = '10.42.0.55'
start_frame_time =0
end_frame_time =0
filtered_packets=[]
features = []

valid_event_values = set(['on','off','active','inactive','open','closed','pushed','held','locked','unlocked','online','offline','hubDisconnected','wet','dry'])

'''
Filters logs from server for the currently analyzed packets' timeframe
'''
def filter_hub_logs():
	global start_frame_time
	global end_frame_time	
	df = pd.read_csv('../hub_logs/logger.csv')
	local = pytz.timezone('US/Eastern')
	start_time_local = local.localize(start_frame_time)
	#print(start_time_local)
	hours_to_add=0
	if bool(start_time_local.dst()):
		hours_to_add = 4
	else:
		hours_to_add = 5
	df['timestamp'] = pd.to_datetime(df['Date/Time'],unit='ns')
	df['timestamp'] +=  pd.to_timedelta(hours_to_add, unit='h')

	start_frame_time = start_frame_time - timedelta(seconds=5)
	end_frame_time = end_frame_time + timedelta(seconds=60)
	df_filtered = df[(df.timestamp >= start_frame_time) & (df.timestamp <= end_frame_time)]
	#df_filtered.to_csv('/home/sk-lab/Desktop/IoTproject/hub_logs/filtered.csv', encoding='utf-8')
	#print zip(df_filtered['timestamp'], df_filtered['Event Name'])
	return  df_filtered

'''
Group packets based on their time difference
'''
def group_packets():
	global filtered_packets

	sequences=[]
	sequence=''
	times=[]

	l = len(filtered_packets)	
	time_delta =10

	for i in range(l-1):
		diff  = filtered_packets[i]['frame']['time']-filtered_packets[i-1]['frame']['time'] 		
		
		'''
		print filtered_packets[i-1]['frame']['time']
		print filtered_packets[i]['frame']['time']
		print filtered_packets[i+1]['frame']['time']		
		print diff.total_seconds()
		#print diff2.total_seconds()
		#print abs(diff.total_seconds() - diff2.total_seconds()) 
		print filtered_packets[i]['frame']['len']
		print '--------------------------------'
		'''
		
		#if the time difference is greater than delta, then it denotes start of a new command 
		if diff.total_seconds() <= time_delta:			
			sequence += filtered_packets[i]['frame']['len']+' '
			###times.append(calendar.timegm(filtered_packets[i]['frame']['time'].timetuple()))
			times.append(filtered_packets[i]['frame']['time'])
				
		else:			
			if len(times)==0:
				times.append(filtered_packets[i]['frame']['time'])
			###sequences.append(( sum(times)/len(times),sequence))
			sequences.append((times[0],sequence ))
			times = [filtered_packets[i]['frame']['time']]	
			sequence=filtered_packets[i]['frame']['len']+' '
	
	return sequences

'''
Group server logs based on their time difference
'''
def group_server_logs(df):
	rows= df.shape[0]
	sequence={}
	sequences=[]
	times =[] 
	time_delta = 10
	#print df['Event Name']
	for ind in range(rows):			
		diff  = (datetime.now()-df.iloc[ind]['timestamp']) if ind==rows-1 else (df.iloc[ind]['timestamp']-df.iloc[ind-1]['timestamp'])	
		dev_key = df.iloc[ind]['Device'].replace(' ','_')
		event_key = df.iloc[ind]['Event Name'].replace(' ','_')
		val = df.iloc[ind]['Event Value']
	
		if diff.total_seconds() <= time_delta :			
			if dev_key in sequence:				
				sequence[dev_key].append({'event':event_key ,'value':val})
			else:
				#sequence[dev_key] = {event_key : val}
				sequence[dev_key]=[{'event':event_key ,'value':val}]
			####times.append( calendar.timegm(df.iloc[ind]['timestamp'].timetuple()))
			times.append(df.iloc[ind]['timestamp'])
		else:			
			if len(times)==0:
				times.append(df.iloc[ind]['timestamp'])
			#sequences.append((sum(times)/len(times),sequence ))
			sequences.append((times[0],sequence ))
			
			times = [df.iloc[ind]['timestamp']]
			sequence = {dev_key:[{'event':event_key ,'value':val}]}
			
	
	return sequences
	
	
'''
Filters packets with data payload and discards acks
'''
def filter_packets(path, file_name):
	global start_frame_time
	global end_frame_time
	global filtered_packets
	global HUB_IP
	packets=[]
	doc = ''
	with open(path+file_name, 'r') as outfile:
     		packets = json.load(outfile)

	for packet in packets:
		if 'tcp' in packet:
			if packet['tcp']['flags']=='0x00000018' and packet.get('ssl')!=None and packet['ssl'].get('app_data')!=None:				
				doc = doc + str((int(packet['frame']['len']) / 100)*100) + ' '
				p = packet
				p['frame']['time'] = datetime.strptime(p['frame']['time'],'%Y-%m-%d %H:%M:%S.%f')
				p['frame']['mod_len']=str((int(packet['frame']['len']) / 100)*100) 
				p['frame']['direction'] = 'hub' if packet['ip']['src'] == HUB_IP else 'server' 
				filtered_packets.append(p)
	start_frame_time =filtered_packets[0]['frame']['time']
	l = len(filtered_packets)
	end_frame_time = filtered_packets[l-1]['frame']['time']
	
	return doc


def convert_timestamp(time_epoch):
	return datetime.strptime( datetime.utcfromtimestamp(float(time_epoch)).strftime('%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')



def get_packet_features():
	global filtered_packets
	global features
	time_delta = 20
	if filtered_packets:
		offset = filtered_packets[0]['frame']['time']
		initial_time = convert_timestamp(filtered_packets[0]['frame']['time_epoch'])
		prev_time = initial_time
		prev_len = filtered_packets[0]['frame']['len']
		for packet in  filtered_packets:
			diff = packet['frame']['time'] - offset
			curr_len = packet['frame']['len']
			if prev_len =='115' and curr_len=='121':
				features.pop()
				prev_len='121'
				continue
			prev_len=curr_len
			if diff.total_seconds() >= time_delta:
				offset = packet['frame']['time']
				diff = 0
			curr_time = convert_timestamp(packet['frame']['time_epoch'])
			feature = {	
						'frame_number'	      : packet['frame']['number'],			 
						'packet_source'       : packet['frame']['direction'],
						'frame_time'	      : packet['frame']['time'],
						'frame_time_epoch'	  : packet['frame']['time_epoch'],
						'frame_length'        : packet['frame']['len'],
						'frame_modlen'	      : packet['frame']['mod_len'],
						'packet_length'       : packet['ip']['len'],
						'first_time_delta'	  : (curr_time - initial_time).total_seconds(),
						'prev_time_delta'	  : (curr_time - prev_time).total_seconds(),
						'offset_time'	      : diff.total_seconds() if diff!=0 else 0.0,
						'sequence_number'	  : -1
					  }
			features.append(feature)
			prev_time = convert_timestamp(features[-1]['frame_time_epoch'])
	

def process_mappings(command,seq):
	commands = []
	
	for device, events in command.items():
		for event in events:
			val = 'XXX'
			command= ''
			if event['value'] in valid_event_values:
				val = event['value']
			command = device#+'_'+event['event']+'_'+val
			all_events.add(command)
			commands.append(command)
	
	return (' '.join(commands),seq)


def part_segments(segment_indexes):
	global features
	segments = {}
	segment_indexes = [0] + segment_indexes+[len(features)]
	sequence_number=1
	for ind in range(1,len(segment_indexes)):
		segment = features[segment_indexes[ind-1]:segment_indexes[ind]]
		if len(segment)>0:
			segments[sequence_number] = segment
			sequence_number+=1
	return segments



def is_time_closer(begin_time, end_time, check_time):
    # If check time is not given, default to current UTC time
	time_offset=10     
	if begin_time <= end_time:
		if check_time >= begin_time and check_time <= end_time:
			return True
		elif abs(begin_time-check_time).total_seconds() <=time_offset or abs(end_time-check_time).total_seconds() <=time_offset:
			return True 
		else:
			return False
	else: # crosses midnight
		return check_time >= begin_time or check_time <= end_time
	

def segment_hub_logs(segments, df_hub_logs):
	global features
	rows= df_hub_logs.shape[0]
	hub_segments = {}

	start_ind=0
	for segment,packets in segments.items():
		seq_start_time = packets[0]['frame_time']
		seq_end_time = packets[-1]['frame_time']
		for ind in range(start_ind,rows):
			event_time = df_hub_logs.iloc[ind]['timestamp']
			dev_key = df_hub_logs.iloc[ind]['Device'].replace(' ','_')
			event_key = df_hub_logs.iloc[ind]['Event Name'].replace(' ','_')
			val = df_hub_logs.iloc[ind]['Event Value']
			hub_data = [{'device':dev_key, 'event': event_key, 'val': val if val in valid_event_values else 'XXX'}]

			if is_time_closer(seq_start_time,seq_end_time,event_time):
				if segment in hub_segments:
					hub_segments[segment] = hub_segments[segment] + hub_data
				else:
					hub_segments[segment]= hub_data
				start_ind = ind+1
			else:				
				break
		frame_lens = ' '.join([p['frame_length'] for p in packets])
		hub_data=[]
		if segment not in hub_segments:
			hub_segments[segment]= hub_data
		if '115 121' in frame_lens:
			hub_data = [{'device':'hub', 'event': 'ping', 'val': 'ping'}]
			hub_segments[segment] = hub_segments[segment] +hub_data
	
	return hub_segments



'''
Maps groups of packets to the groups of cluster logs
'''
def generate_mappings(path,file_name):

	global filtered_packets
	global features 
	features = []
	filtered_packets=[]
	final_segments = []
	packet_time_deltas = []
	start_ind=0
	time_delta_threshold = 25

	filter_packets(path,file_name)
	get_packet_features()

	
	flag=0
	frame_lens = ''
	for ind,feature in enumerate(features):
		frame_lens = frame_lens +' '+ feature['frame_length']
		if  feature['prev_time_delta'] > time_delta_threshold:
			if flag>0 and len(packet_time_deltas)>10:
				segments = segment_packets(packet_time_deltas)
				prev_segment_ind = segments[-1]	
				if start_ind!=0:
					segments = [start_ind+x for x in segments]
				final_segments = final_segments + segments
				start_ind=ind- (len(packet_time_deltas)-prev_segment_ind)
				packet_time_deltas = packet_time_deltas[prev_segment_ind:]
				flag=0
				frame_lens = ''
			flag=flag+1
		packet_time_deltas.append(feature['first_time_delta'])
	
	if packet_time_deltas:
		segments = segment_packets(packet_time_deltas)
		if start_ind!=0:
				segments = [start_ind+x-1 for x in segments][1:]
		final_segments = final_segments + segments

	pcap_segments = part_segments(final_segments)	
	df_filtered_hub_logs = filter_hub_logs()
	hub_segments = segment_hub_logs(pcap_segments, df_filtered_hub_logs)

	with open('../results/train/pcap_segments/'+file_name,'w') as out:
		out.write(json.dumps(pcap_segments, sort_keys=False,indent=4,ensure_ascii=False, default=str))

	with open('../results/train/hub_segments/'+file_name,'w') as out:
		out.write(json.dumps(hub_segments, sort_keys=False,indent=4,ensure_ascii=False, default=str))
	print(file_name)

	'''
	seq_1 = group_packets()
	seq_2= group_server_logs(filter_hub_logs())
	command_seq=[]
	'''
	'''
	for (time,seq) in seq_2:
		for i in range(len(seq_1)):
			t2,_= seq_1[i]
			#loop until two groups with minimum time difference between them is found and stop searching for more groups
			if abs(t2-time)<=20:	
				
				if command_seq:
					prev_cmd,prev_seq = command_seq[-1]
					prev_seq = prev_seq + ' '.join([seq_1[x][1].rstrip() for x in range(0,i)])
					command_seq[-1] = (prev_cmd,prev_seq) 		
					
				command_seq.append(process_mappings(seq,seq_1[i][1]))				
				seq_1 = seq_1[i+1:]
				#seq_1.pop(i)
				break
	'''
	'''
	for (time,seq) in seq_2:
		mapped=[]
		per_seq = ''
		for i in range(len(seq_1)):
			t2,_= seq_1[i]
			#loop until two groups with minimum time difference between them is found and stop searching for more groups
			if abs((t2-time).total_seconds())<=20:	
				#keep it commented
				if command_seq:
					prev_cmd,prev_seq = command_seq[-1]
					prev_seq = prev_seq + ' '.join([seq_1[x][1].rstrip() for x in range(0,i)])
					command_seq[-1] = (prev_cmd,prev_seq) 		
					
				
				per_seq +=seq_1[i][1]
				mapped.append(i)
			elif len(per_seq)>0:
				break
		if per_seq=='' and len(command_seq)>0:
			cmd,_ = process_mappings(seq,per_seq)
			command_seq[-1] = (command_seq[-1][0]+' '+cmd,command_seq[-1][1])	
		else:
			if "115 121 " in per_seq:
				per_seq = per_seq.replace("115 121 ","")
			command_seq.append(process_mappings(seq,per_seq))
		if mapped:
			seq_1=seq_1[0:mapped[0]]+seq_1[mapped[-1]+1:]
	'''
	'''
	print '************unmapped packets*******************'
	print seq_1
	'''
	#return command_seq

import os 
commands = []
sequences = []
command_seq = []
command_seq2={}
all_events = set()
path = '/home/sk-lab/Desktop/IoTproject/json/train/'

for file in  os.listdir(path):
	if file!='logger.json':# and '1motion' in file:
		generate_mappings(path,file)
	'''
		for com_seq in generate_mappings(path,file):
			commands.append(com_seq[0])#+'\n'+com_seq[1]+'\n------------------------------------------------------------------------------------')
			sequences.append(com_seq[1])
			com = list(set(com_seq[0].split(' ')))[0]
			if command_seq2.get(com)!=None:
				command_seq2[com] = command_seq2[com] + com_seq[1]
			else:
				command_seq2[com]=com_seq[1]
			command_seq.append({'device':com_seq[0],'sequence':com_seq[1].rstrip().split(' ')})
	'''	

'''
with open('../results/command_mapping_train.txt','w') as out:
	out.write('\n'.join(commands))

with open('../results/seq_mapping_train.txt','w') as out:
	out.write('\n'.join(sequences))

with open('../results/events_train.txt','w') as out:
	out.write('\n'.join(all_events))


df_features = pd.DataFrame(features)
df_features.to_csv('../results/features.csv')
'''
'''
with open('command_seq_mapping_HMM_input_vect_test.txt','w') as out:
	 out.write(json.dumps(command_seq2, sort_keys=False,indent=4,encoding="utf-8",ensure_ascii=False))



with open('command_seq_mapping_HMM_input_test.txt','w') as out:
	 out.write(json.dumps(command_seq, sort_keys=False,indent=4,encoding="utf-8",ensure_ascii=False))


'''


'''
def map_commands_to_packets(df):
	global filtered_packets
	l = len(filtered_packets)
	count=0
	#print df
	sequences=[]
	rows= df.shape[0]
	for ind in range(rows):
		sequence=''
		print '************************************************************'
		print df.iloc[ind]['Event Name']
		print df.iloc[ind]['Event Value']
		for i in range(count,l):
			diff  = df.iloc[ind]['timestamp']-filtered_packets[i]['frame']['time'] 			
			diff2 =  (datetime.now()-filtered_packets[i]['frame']['time']) if ind==rows-1 else (df.iloc[ind+1]['timestamp']-filtered_packets[i]['frame']['time'])
			
			print df.iloc[ind]['timestamp']
			print filtered_packets[i]['frame']['time']
			print df.iloc[ind+1]['timestamp']
			print filtered_packets[i]['frame']['time']
			print diff.total_seconds()
			print diff2.total_seconds()
			print abs(diff.total_seconds()) < abs(diff2.total_seconds()) 
			print filtered_packets[i]['frame']['len']
			print '--------------------------------'
			
			if  abs(diff.total_seconds()) <= abs(diff2.total_seconds())  :
				#print filtered_packets[i]['frame']['time']
				sequence += filtered_packets[i]['frame']['mod_len']+' '
			else:	
				count=i			
				break
		print sequence
		print df.iloc[ind]['timestamp']
		print df.iloc[ind]['Event Name']
		print df.iloc[ind]['Event Value']
		sequences.append((df.iloc[ind]['Event Name'],sequence))
	print sequences

Ignore this method!!!

def get_frequency_count():
	corpus = []
	corpus.append(filter_packet_lengths())
	vectorizer = CountVectorizer(ngram_range=(2,5))
	X = vectorizer.fit_transform(corpus)
	vocab = vectorizer.vocabulary_
	count_values = X.toarray().sum(axis=0)
	counts = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
	#print(counts[:30])
'''
